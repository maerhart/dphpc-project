#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include <clang/AST/Decl.h>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory ConverterCategory("GPU MPI converter options");

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

static cl::extrahelp MoreHelp("\nMore help...\n");

const char* GPU_MPI_PROJECT = "GPU_MPI_PROJECT";
const char* GPU_MPI_MAX_RANKS = "GPU_MPI_MAX_RANKS";

int getMaxRanks() {
    int res = 1024;

    char* maxRanks = getenv(GPU_MPI_MAX_RANKS);
    if (maxRanks) {
        res = atoi(maxRanks);
        if (res <= 0) {
            llvm::errs() << "ERROR: " << GPU_MPI_MAX_RANKS << " environment variable should contain number of ranks!";
            exit(1);
        }
    }

    return res;
}

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

                // Insert function to query maximum supported number of threads
                mRewriter.InsertTextAfterToken(func->getEndLoc(),
                    "\n"
                    "__device__ int __gpu_max_threads() {\n"
                    "    return " + std::to_string(getMaxRanks()) + ";\n"
                    "}\n"
                );
            }
            //llvm::errs() << "Annotating function " << func->getName() << " with __device__\n";
            SourceLocation unexpandedLocation = func->getSourceRange().getBegin();
            SourceLocation expandedLocation = mRewriter.getSourceMgr().getFileLoc(unexpandedLocation);
            bool error = mRewriter.InsertTextBefore(expandedLocation, "__device__ ");
            assert(!error);

            // Check for old style function parameter list and fix it.
            // For example "void foo(a, b) int a; float b; {}" -> "void foo(int a, float b) {}"
            FunctionTypeLoc ftl = func->getFunctionTypeLoc();

            SourceLocation typedParamsStart;
            SourceLocation typedParamsEnd;
            std::string fixedParameterString;
            for (unsigned i = 0; i < func->getNumParams(); i++) {
                const ParmVarDecl* paramDecl = func->getParamDecl(i);
                if (!ftl.getParensRange().fullyContains(paramDecl->getSourceRange())) {
                    // we found it!
                    QualType paramType = paramDecl->getType();
                    std::string typeStr = paramType.getAsString();
                    std::string nameStr = paramDecl->getNameAsString();
                    if (i != 0) {
                        fixedParameterString += ", ";
                    }
                    fixedParameterString += typeStr + " " + nameStr;

                    SourceLocation startLocation = paramDecl->getBeginLoc();
                    SourceLocation endLocation = paramDecl->getEndLoc().getLocWithOffset(1); // account for semicolon

                    if (typedParamsStart.isInvalid() || mRewriter.getSourceMgr().isBeforeInTranslationUnit(startLocation, typedParamsStart)) {
                        typedParamsStart = startLocation;
                    }
                    if (typedParamsEnd.isInvalid() || mRewriter.getSourceMgr().isBeforeInTranslationUnit(typedParamsEnd, endLocation)) {
                        typedParamsEnd = endLocation;
                    }
                }

            }
            if (!fixedParameterString.empty()) {
                mRewriter.ReplaceText(ftl.getParensRange(), "(" + fixedParameterString + ")");
                mRewriter.RemoveText(SourceRange(typedParamsStart, typedParamsEnd));
            }
        }
        if (const ImplicitCastExpr *ice = Result.Nodes.getNodeAs<ImplicitCastExpr>("implicitCast")) {
            if (ice->getCastKind() == CastKind::CK_BitCast) {
                mRewriter.InsertTextBefore(ice->getSourceRange().getBegin(), "(" + ice->getType().getAsString() + ")");
            }
        }
        if (const VarDecl *var = Result.Nodes.getNodeAs<VarDecl>("globalVar")) {
            int maxRanks = getMaxRanks();

            mRewriter.InsertTextBefore(var->getSourceRange().getBegin(), "__device__ ");
            SourceLocation varNameLocation = var->getLocation();
            mRewriter.InsertTextAfterToken(varNameLocation, "[" + std::to_string(maxRanks) + "]");

            if (var->hasInit()) {
                if (var->getInitStyle() != VarDecl::InitializationStyle::CInit) {
                    llvm::errs() << "ERROR: only C-style initialization is supported for variables with global storage\n";
                    exit(1);
                }

                const Expr* initializer = var->getAnyInitializer();

                initializer->getSourceRange();
                std::string initializerText = Lexer::getSourceText(CharSourceRange::getTokenRange(initializer->getSourceRange()), mRewriter.getSourceMgr(), mRewriter.getLangOpts()).str();
                std::string newInitText = "{" + initializerText;
                for (int i = 1; i < maxRanks; i++) {
                    newInitText += ", " + initializerText;
                }
                newInitText += "}";

                mRewriter.ReplaceText(initializer->getSourceRange(), newInitText);

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

            // check if we already applied this rewrite
            std::string rewrittenText = mRewriter.getRewrittenText(SourceRange(beginLoc, beginLoc));
            if (rewrittenText.find("__gpu_global") == std::string::npos) {
                // we didn't apply rewrite yet, so we proceed
                mRewriter.InsertTextAfter(beginLoc, "__gpu_global("); // we need to insert it AFTER previous insertion to avoid issue with implicit cast before
                mRewriter.InsertTextAfterToken(endLoc, ")");
            }

        }
        if (const DeclRefExpr* refClassTokenDecl = Result.Nodes.getNodeAs<DeclRefExpr>("refClassTokenDecl")) {
            mRewriter.InsertText(refClassTokenDecl->getSourceRange().getBegin(), "__decl_");
        }
        if (const NamedDecl* classTokenDecl = Result.Nodes.getNodeAs<NamedDecl>("classTokenDecl")) {
            mRewriter.InsertText(classTokenDecl->getLocation(), "__decl_");
        }
        if (const CallExpr *mallocCall = Result.Nodes.getNodeAs<CallExpr>("mallocCallInMain")) {
            mRewriter.InsertTextBefore(mallocCall->getRParenLoc(), ", true");
            // SourceRange oldMallocRange = mallocCall->getCallee()->getSourceRange();
            // mRewriter.ReplaceText(oldMallocRange, "dyn_malloc");
        }
        if (const CallExpr *mallocCall = Result.Nodes.getNodeAs<CallExpr>("mallocCall")) {
            mRewriter.InsertTextBefore(mallocCall->getRParenLoc(), ", __coalesced");
            // SourceRange oldMallocRange = mallocCall->getCallee()->getSourceRange();
            // mRewriter.ReplaceText(oldMallocRange, "dyn_malloc");
        }
        if (const CallExpr *mallocCall = Result.Nodes.getNodeAs<CallExpr>("callToAddArg")) {
            mRewriter.InsertTextBefore(mallocCall->getRParenLoc(), ", __coalesced");
        }
        if (const CallExpr *mallocCall = Result.Nodes.getNodeAs<CallExpr>("callToAddTrue")) {
            mRewriter.InsertTextBefore(mallocCall->getRParenLoc(), ", true");
        }
        if (const FunctionDecl *func = Result.Nodes.getNodeAs<FunctionDecl>("funcToAddArg")) {
            SourceLocation paramEnd = func->getFunctionTypeLoc().getRParenLoc();
            mRewriter.InsertTextAfter(paramEnd, ", bool __coalesced = false");
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

        mMatcher.addMatcher(implicitCastExpr().bind("implicitCast"), &mFuncConverter);

        // match both global variables and variables inside functions with "static" specifier
        mMatcher.addMatcher(varDecl(hasGlobalStorage()).bind("globalVar"), &mFuncConverter);

        // match references to global vars
        mMatcher.addMatcher(declRefExpr(to(varDecl(hasGlobalStorage()))).bind("refToGlobalVar"), &mFuncConverter);

        // in C it is possible to declare variable with name "class". For example: "int class = 0;"
        // CUDA will not accept it, so we need to rename such occurences.
        mMatcher.addMatcher(declRefExpr(to(namedDecl(hasName("class")))).bind("refClassTokenDecl"), &mFuncConverter);
        mMatcher.addMatcher(namedDecl(hasName("class")).bind("classTokenDecl"), &mFuncConverter);

        mMatcher.addMatcher(functionDecl(isMain(), unless(isImplicit()),
            forEachDescendant(callExpr(
                unless(anyOf(hasAncestor(ifStmt()), hasAncestor(forStmt()), hasAncestor(whileStmt()), hasAncestor(doStmt()), hasAncestor(conditionalOperator()))),
                callee(functionDecl(hasName("malloc")))).bind("mallocCallInMain"))), &mFuncConverter);

        mMatcher.addMatcher(functionDecl(unless(isMain()), unless(isImplicit()),
            forEachDescendant(callExpr(
                unless(anyOf(hasAncestor(ifStmt()), hasAncestor(forStmt()), hasAncestor(whileStmt()), hasAncestor(doStmt()), hasAncestor(conditionalOperator()))),
                callee(functionDecl(hasName("malloc")))).bind("mallocCall"))), &mFuncConverter);

        mMatcher.addMatcher(functionDecl(unless(isMain()), unless(isImplicit()),
            hasDescendant(callExpr(
                unless(anyOf(hasAncestor(ifStmt()), hasAncestor(forStmt()), hasAncestor(whileStmt()), hasAncestor(doStmt()), hasAncestor(conditionalOperator()))),
                callee(functionDecl(hasName("malloc")))))).bind("funcToAddArg"), &mFuncConverter);

        mMatcher.addMatcher(functionDecl(unless(isMain()), unless(isImplicit()),
            forEachDescendant(callExpr(
                unless(anyOf(hasAncestor(ifStmt()), hasAncestor(forStmt()), hasAncestor(whileStmt()), hasAncestor(doStmt()), hasAncestor(conditionalOperator()))),
                callee(functionDecl(hasDescendant(callExpr(
                unless(anyOf(hasAncestor(ifStmt()), hasAncestor(forStmt()), hasAncestor(whileStmt()), hasAncestor(doStmt()), hasAncestor(conditionalOperator()))),
                callee(functionDecl(hasName("malloc")))))))).bind("callToAddArg"))), &mFuncConverter);

        mMatcher.addMatcher(functionDecl(isMain(), unless(isImplicit()),
            forEachDescendant(callExpr(
                unless(anyOf(hasAncestor(ifStmt()), hasAncestor(forStmt()), hasAncestor(whileStmt()), hasAncestor(doStmt()), hasAncestor(conditionalOperator()))),
                callee(functionDecl(hasDescendant(callExpr(
                unless(anyOf(hasAncestor(ifStmt()), hasAncestor(forStmt()), hasAncestor(whileStmt()), hasAncestor(doStmt()), hasAncestor(conditionalOperator()))),
                callee(functionDecl(hasName("malloc")))))))).bind("callToAddTrue"))), &mFuncConverter);
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

            // add headers to support global variable handling
            SourceLocation fileStart = mRewriter.getSourceMgr().translateFileLineCol(fileEntry, /*line*/1, /*column*/1);
            mRewriter.InsertTextBefore(fileStart, "#include \"global_vars.cuh\"\n"); // this header required to make __gpu_global function available in user code

            llvm::errs() << "Trying to write " << fileName << " : " << fileEntry->tryGetRealPathName() << "\n";

            std::string newFileName;
            if (fileID == mRewriter.getSourceMgr().getMainFileID()) {
                size_t lastDotIdx = fileName.rfind(".");
                assert(lastDotIdx != std::string::npos);
                std::string fileNameBase = fileName.str().substr(0, lastDotIdx);
                newFileName = fileNameBase + ".cu";
            } else {
                newFileName = fileName.str() + ".cuh";
            }

            std::error_code error_code;
            raw_fd_ostream outFile(newFileName, error_code, llvm::sys::fs::OF_None);
            assert(!outFile.has_error());
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
