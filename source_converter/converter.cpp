#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "llvm/Support/CommandLine.h"

#include <map>
#include <fstream>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

/* code below is copied from "clang/lib/ARCMigrate/Transforms.h" */
 
/// \arg Loc is the end of a statement range. This returns the location
/// of the semicolon following the statement.
/// If no semicolon is found or the location is inside a macro, the returned
/// source location will be invalid.
SourceLocation findSemiAfterLocation(SourceLocation loc,
                                            ASTContext &Ctx,
                                            bool IsDecl) {
  SourceManager &SM = Ctx.getSourceManager();
  if (loc.isMacroID()) {
    if (!Lexer::isAtEndOfMacroExpansion(loc, SM, Ctx.getLangOpts(), &loc))
      return SourceLocation();
  }
  loc = Lexer::getLocForEndOfToken(loc, /*Offset=*/0, SM, Ctx.getLangOpts());

  // Break down the source location.
  std::pair<FileID, unsigned> locInfo = SM.getDecomposedLoc(loc);

  // Try to load the file buffer.
  bool invalidTemp = false;
  StringRef file = SM.getBufferData(locInfo.first, &invalidTemp);
  if (invalidTemp)
    return SourceLocation();

  const char *tokenBegin = file.data() + locInfo.second;

  // Lex from the start of the given location.
  Lexer lexer(SM.getLocForStartOfFile(locInfo.first),
              Ctx.getLangOpts(),
              file.begin(), tokenBegin, file.end());
  Token tok;
  lexer.LexFromRawLexer(tok);
  if (tok.isNot(tok::semi)) {
    if (!IsDecl)
      return SourceLocation();
    // Declaration may be followed with other tokens; such as an __attribute,
    // before ending with a semicolon.
    return findSemiAfterLocation(tok.getLocation(), Ctx, /*IsDecl*/true);
  }

  return tok.getLocation();
}

/// 'Loc' is the end of a statement range. This returns the location
/// immediately after the semicolon following the statement.
/// If no semicolon is found or the location is inside a macro, the returned
/// source location will be invalid.
SourceLocation findLocationAfterSemi(SourceLocation loc,
                                            ASTContext &Ctx, bool IsDecl) {
  SourceLocation SemiLoc = findSemiAfterLocation(loc, Ctx, IsDecl);
  if (SemiLoc.isInvalid())
    return SourceLocation();
  return SemiLoc.getLocWithOffset(1);
}

/* code above is copied from "clang/lib/ARCMigrate/Transforms.h" */


// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...\n");

class FuncConverter : public MatchFinder::MatchCallback {
public :
    FuncConverter(Rewriter& rewriter) : mRewriter(rewriter) {

    }
    virtual void run(const MatchFinder::MatchResult &Result) {
        if (const FunctionDecl *func = Result.Nodes.getNodeAs<FunctionDecl>("func")) {
            if (func->isMain()) {
                llvm::errs() << "Detected main function\n";

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
            
            // check that global vars comes from user header, not system header
            if (!srcMgr.isInSystemHeader(beginLoc)) {
                //llvm::errs() << refToGlobalVar << " source location: " << originalSrcRange.printToString(srcMgr) << " " << beginLoc.printToString(srcMgr) << "," << endLoc.printToString(srcMgr) << "\n";
                std::string text = mRewriter.getRewrittenText(SourceRange(beginLoc, endLoc));
                // make sure that such text is not already added, for example, inside the macro
                if (text.rfind("__gpu_global") != 0) { // search for prefix
                    const VarDecl* vd = dynamic_cast<const VarDecl*>(refToGlobalVar->getDecl());
                    assert(vd);
                    int vdid = getVarDeclId(vd);

                    // we need to insert it AFTER previous insertion to avoid issue with implicit cast before
                    mRewriter.InsertTextAfter(beginLoc, "__gpu_global<" + std::to_string(vdid) + ">(");
                    mRewriter.InsertTextAfterToken(endLoc, ")");
                }
            }
        }
    }
    void emitGlobalVars() {
        std::ofstream fsc("gpumpi_globals.cu");

        assert(fsc.is_open());

        fsc << "__device__ int __gpu_num_globals = " + std::to_string(declMap.size()) + ";\n";
        fsc << "__device__ char *__gpu_global_ptrs[" + std::to_string(declMap.size()) + "] = {0};\n";
        fsc << "__device__ int __gpu_global_size[" + std::to_string(declMap.size()) + "] = {0};\n";

        for (auto pair : declMap) {
            const VarDecl* vd = pair.first;
            int idx = pair.second;
            std::string idx_str = std::to_string(idx);
            if (!vd->isStaticLocal()) {
                std::string initFuncName = "__gpu_init_global_" + idx_str;
                fsc << "__device__ void " + initFuncName + "();\n";
            }
        }

        fsc << "__device__ void __gpu_init_global_ptrs() {\n";
        for (auto pair : declMap) {
            const VarDecl* vd = pair.first;
            int idx = pair.second;
            std::string idx_str = std::to_string(idx);

            SourceLocation globalVarDeclEnd = findLocationAfterSemi(vd->getEndLoc(), vd->getASTContext(), true); 
            std::string globalVarName = vd->getNameAsString();
            //std::string typeStr = QualType::getAsString(vd->getType().split(), PrintingPolicy{{}}); // may be useful later


            if (vd->isStaticLocal()) {
                mRewriter.InsertText(globalVarDeclEnd, "\n__gpu_init_global<" + idx_str + ">(" + globalVarName + ");\n");
            } else {
                std::string initFuncName = "__gpu_init_global_" + idx_str;

                fsc << initFuncName + "();\n";

                mRewriter.InsertText(globalVarDeclEnd, 
                    "\n__device__ void " + initFuncName + "() {\n"
                    "__gpu_global_ptrs[" + idx_str + "] = &" + globalVarName + ";\n"
                    "__gpu_global_size[" + idx_str + "] = sizeof(" + globalVarName + ");\n"
                    "}\n");
            }
            
        }

        fsc << "}\n";
    }

private:
    int getVarDeclId(const VarDecl* vd) {
        assert(vd);
        vd = vd->getCanonicalDecl();

        // we will use pointer to vd to uniquely identify var to which we refer
        auto it = declMap.find(vd);
        if (it == declMap.end()) {
            bool success = false;
            std::tie(it, success) = declMap.insert(std::make_pair(vd, int(declMap.size())));
            assert(success);
        }

        int varDeclId = it->second;
        return varDeclId;
    }

    std::map<const VarDecl*, int> declMap;
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
        
        // write file with initialization of detected global variables
        mFuncConverter.emitGlobalVars();
    }

private:
    FuncConverter mFuncConverter;
    MatchFinder mMatcher;
};

class MyFrontendAction : public ASTFrontendAction {
public:
    void EndSourceFileAction() override {
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

            // silly detection of system headers:
            // if name starts with '/' then it is system header
            if (fileName[0] == '/') {
                //llvm::errs() << "Skip " << fileName << "\n";
                continue; // skip system headers
            }

            // add headers to support global variable handling
            SourceLocation fileStart = mRewriter.getSourceMgr().translateFileLineCol(fileEntry, /*line*/1, /*column*/1);
            mRewriter.InsertTextBefore(fileStart, "#include \"global_vars.cuh\"\n");

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
const char* extra_arg = "-extra-arg=-I" STR(LLVM_BUILTIN_HEADERS);
#undef STR
#undef STR2

int main(int argc, const char **argv) {
    int adj_argc = argc + 1;
    std::vector<const char*> adj_argv(adj_argc);
    adj_argv[0] = argv[0];
    adj_argv[1] = extra_arg;
    for (int i = 1; i < argc; i++) {
        adj_argv[i + 1] = argv[i];
    }
    
    CommonOptionsParser OptionsParser(adj_argc, adj_argv.data(), MyToolCategory);
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());
    return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
