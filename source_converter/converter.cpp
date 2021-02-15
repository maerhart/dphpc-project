#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory ConverterCategory("GPU MPI converter options");

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

static cl::extrahelp MoreHelp("\nMore help...\n");

const char* GPU_MPI_PROJECT = "GPU_MPI_PROJECT";

class FuncConverter : public MatchFinder::MatchCallback {
public :
    FuncConverter(Rewriter& rewriter) : mRewriter(rewriter) {

    }
    virtual void run(const MatchFinder::MatchResult &Result) {
        if (const FunctionDecl *func = Result.Nodes.getNodeAs<FunctionDecl>("func")) {
            if (func->isMain()) {
                //llvm::errs() << "Detected main function\n";

                mRewriter.ReplaceText(func->getNameInfo().getSourceRange(), "__gpu_main");

                clang::TypeLoc tl = func->getTypeSourceInfo()->getTypeLoc();
                clang::FunctionTypeLoc ftl = tl.getAsAdjusted<FunctionTypeLoc>();
                mRewriter.ReplaceText(ftl.getParensRange(), "(int argc, char** argv)");
            }
            //llvm::errs() << "Annotating function " << func->getName() << " with __device__\n";
            SourceLocation unexpandedLocation = func->getSourceRange().getBegin();
            SourceLocation expandedLocation = mRewriter.getSourceMgr().getFileLoc(unexpandedLocation);
            bool error = mRewriter.InsertTextBefore(expandedLocation, "__device__ ");
            assert(!error);
        }
        if (const VarDecl *var = Result.Nodes.getNodeAs<VarDecl>("globalVar")) {
            //llvm::errs() << "GLOBAL VAR!!!\n";
            mRewriter.InsertTextBefore(var->getSourceRange().getBegin(), "__device__ ");
        }
        if (const ImplicitCastExpr *ice = Result.Nodes.getNodeAs<ImplicitCastExpr>("implicitCast")) {
            if (ice->getCastKind() == CastKind::CK_BitCast) {
                mRewriter.InsertTextBefore(ice->getSourceRange().getBegin(), "(" + ice->getType().getAsString() + ")");
            }
        }
        if (const DeclRefExpr* refToGlobalVar = Result.Nodes.getNodeAs<DeclRefExpr>("refToGlobalVar")) {
            // WARNING! for some reason the same refToGlobalVar can be matched multiple times,
            // it could be bug in libTooling or some feature that I miss.
            // It happens for references to const global variables in the global scope to define other global variables.
            
            // enclose access by __gpu_global( ... ) annotation
            SourceManager& srcMgr = mRewriter.getSourceMgr();
            SourceRange originalSrcRange = refToGlobalVar->getSourceRange();
            SourceLocation beginLoc = srcMgr.getSpellingLoc(originalSrcRange.getBegin());
            SourceLocation endLoc = srcMgr.getSpellingLoc(originalSrcRange.getEnd());
            //llvm::errs() << refToGlobalVar << " source location: " << originalSrcRange.printToString(srcMgr) << " " << beginLoc.printToString(srcMgr) << "," << endLoc.printToString(srcMgr) << "\n";
            mRewriter.InsertTextAfter(beginLoc, "__gpu_global("); // we need to insert it AFTER previous insertion to avoid issue with implicit cast before
            mRewriter.InsertTextAfterToken(endLoc, ")");
        }
    }
private:
    Rewriter& mRewriter;
};

class MyASTConsumer : public ASTConsumer {
public:
    MyASTConsumer(Rewriter &rewriter)
        : mFuncConverter(rewriter) 
    {
        // Match only explcit function declarations (that are written by user, but not
        // added with compiler). This helps to avoid looking at builtin functions.
        // Since implicit constructors in C++ also require __device__ annotation,
        // we can't support them and stick to supporting only C subset.
        mMatcher.addMatcher(functionDecl(unless(isImplicit())).bind("func"), &mFuncConverter);

        mMatcher.addMatcher(varDecl(hasGlobalStorage(), unless(isStaticLocal())).bind("globalVar"), &mFuncConverter);

        mMatcher.addMatcher(implicitCastExpr().bind("implicitCast"), &mFuncConverter);

        mMatcher.addMatcher(declRefExpr(to(varDecl(hasGlobalStorage()))).bind("refToGlobalVar"), &mFuncConverter);
    }

    void HandleTranslationUnit(ASTContext &Context) override {
        // Run the matchers when we have the whole TU parsed.
        mMatcher.matchAST(Context);

    }

private:
    FuncConverter mFuncConverter;
    MatchFinder mMatcher;
};

class MyFrontendAction : public ASTFrontendAction {
public:
    void EndSourceFileAction() override {
        // create directory to make a copy of source tree with device annotations
        

        for (auto I = mRewriter.buffer_begin(), E = mRewriter.buffer_end(); I != E; ++I) {
            FileID fileID = I->first;
            RewriteBuffer& rb = I->second;
            if (fileID.isInvalid()) {
                llvm::errs() << "fileID == 0\n";
                continue;
            }

            const FileEntry *fileEntry = mRewriter.getSourceMgr().getFileEntryForID(fileID);
            assert(fileEntry);
            StringRef fileName = fileEntry->getName();

            // detect file location, if it is outside project dir, skip its processing 
            char* projectDirC = getenv(GPU_MPI_PROJECT);
            assert(projectDirC);
            projectDirC = realpath(projectDirC, NULL);
            assert(projectDirC);
            std::string projectDir(projectDirC);
            free(projectDirC);

            char* filePathC = realpath(fileName.str().c_str(), NULL);
            assert(filePathC);
            std::string filePath(filePathC);
            free(filePathC);
            
            if (0 != filePath.rfind(projectDir, 0)) {
                //llvm::errs() << "Skip " << fileName << " because it is not in project dir\n";
                continue;
            }

            llvm::errs() << "Trying to write " << fileName << " : " << fileEntry->tryGetRealPathName() << "\n";

            std::string fileExtension;
            if (fileID == mRewriter.getSourceMgr().getMainFileID()) {
                fileExtension = ".cu";
            } else {
                fileExtension = ".cuh";
            }

            std::error_code error_code;
            raw_fd_ostream outFile((fileName + fileExtension).str(), error_code, llvm::sys::fs::OF_None);
            mRewriter.getEditBuffer(fileID).write(outFile);
        }
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(
            CompilerInstance& ci, StringRef file) override
    {
        llvm::errs() << "** Creating AST consumer for: " << file << "\n";
        mRewriter.setSourceMgr(ci.getSourceManager(), ci.getLangOpts());
        return std::make_unique<MyASTConsumer>(mRewriter);
    }
private:
    Rewriter mRewriter;
};

#define STR2(x) #x
#define STR(x) STR2(x)
const char* isystem = "-isystem";
const char* builtin_headers = STR(LLVM_BUILTIN_HEADERS);
#undef STR
#undef STR2

int main(int argc, const char **argv) {
    // add system headers from local llvm installation
    std::vector<const char*> adj_argv;
    for (int i = 0; i < argc; i++) {
        adj_argv.push_back(argv[i]);
    }
    adj_argv.push_back(isystem);
    adj_argv.push_back(builtin_headers);
    int adj_argc = adj_argv.size();
    
    // check that project dir is specified before proceeding
    char* projectDir = getenv(GPU_MPI_PROJECT);
    if (!projectDir) {
        llvm::errs() << "ERROR! " << GPU_MPI_PROJECT << " environment variable is not specified. You should specify it before running converter!\n";
        return 1;
    }
    char* projectDirRealPath = realpath(projectDir, NULL);
    if (!projectDirRealPath) {
        llvm::errs() << "ERROR! " << GPU_MPI_PROJECT << " doesn't point to existing file!\n";
        return 1;
    }
    free(projectDirRealPath);


    CommonOptionsParser OptionsParser(adj_argc, adj_argv.data(), ConverterCategory);
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());
    return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
