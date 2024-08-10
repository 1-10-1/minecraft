#include <mc/asserts.hpp>
#include <mc/renderer/backend/shader.hpp>
#include <mc/utils.hpp>

#include <atomic>
#include <iostream>
#include <ranges>
#include <vector>

#include <glslang/SPIRV/GlslangToSpv.h>
#include <glslang/SPIRV/disassemble.h>

// Findings from inspecting standalone.cpp (glslang repository)
//
// https://chatgpt.com/share/de9c0424-aab0-4140-ac15-9d4b386d74b6
//
// It is possible to override binding and set indices using the four functions:
//     1) ProcessBindingBase
//     2) ProcessResourceSetBindingBase
//     3) ProcessBlockStorage
//     4) ProcessGlobalBlockSettings
// Go crawling around the codebase to figure out how they're used!

namespace
{
    // These shouldn't be hanging out here
    std::atomic<int8_t> compileFailed { 0 };
    std::atomic<int8_t> linkFailed { 0 };
    std::atomic<int8_t> compileOrLinkFailed { 0 };

    using namespace renderer::backend;

    // Writes a string into a depfile, escaping some special characters following the Makefile rules.
    static void writeEscapedDepString(std::ofstream& file, std::string const& str)
    {
        for (char c : str)
        {
            switch (c)
            {
                case ' ':
                case ':':
                case '#':
                case '[':
                case ']':
                case '\\':
                    file << '\\';
                    break;
                case '$':
                    file << '$';
                    break;
            }
            file << c;
        }
    }

    // Writes a depfile similar to gcc -MMD foo.c
    bool writeDepFile(std::string depfile,
                      std::vector<std::string>& binaryFiles,
                      std::vector<std::string> const& sources)
    {
        std::ofstream file(depfile);
        if (file.fail())
            return false;

        for (auto binaryFile = binaryFiles.begin(); binaryFile != binaryFiles.end(); binaryFile++)
        {
            writeEscapedDepString(file, *binaryFile);
            file << ":";
            for (auto sourceFile = sources.begin(); sourceFile != sources.end(); sourceFile++)
            {
                file << " ";
                writeEscapedDepString(file, *sourceFile);
            }
            file << std::endl;
        }
        return true;
    }

    struct ShaderCompUnit
    {
        EShLanguage stage;
        static int const maxCount = 1;
        int count;                           // live number of strings/names
        char const* text[maxCount];          // memory owned/managed externally
        std::string fileName[maxCount];      // hold's the memory, but...
        char const* fileNameList[maxCount];  // downstream interface wants pointers

        ShaderCompUnit(EShLanguage stage) : stage(stage), count(0) {}

        ShaderCompUnit(ShaderCompUnit const& rhs)
        {
            stage = rhs.stage;
            count = rhs.count;
            for (int i = 0; i < count; ++i)
            {
                fileName[i]     = rhs.fileName[i];
                text[i]         = rhs.text[i];
                fileNameList[i] = rhs.fileName[i].c_str();
            }
        }

        void addString(std::string& ifileName, char const* itext)
        {
            assert(count < maxCount);
            fileName[count]     = ifileName;
            text[count]         = itext;
            fileNameList[count] = fileName[count].c_str();
            ++count;
        }
    };

    void CompileAndLinkShaderUnits(std::vector<ShaderCompUnit> compUnits)
    {
        // keep track of what to free
        std::list<glslang::TShader*> shaders;

        EShMessages messages = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules);

        // you can use includer.pushExternalLocalDirectory(...) to add include directories
        DirStackFileIncluder includer;

        std::vector<std::string> sources;

        //
        // Per-shader processing...
        //

        glslang::TProgram program;

        for (auto it = compUnits.cbegin(); it != compUnits.cend(); ++it)
        {
            auto const& compUnit = *it;

            for (int i = 0; i < compUnit.count; i++)
            {
                sources.push_back(compUnit.fileNameList[i]);
            }

            glslang::TShader* shader = new glslang::TShader(compUnit.stage);

            shader->setStringsWithLengthsAndNames(
                compUnit.text, nullptr, compUnit.fileNameList, compUnit.count);

            // NOTE(aether) this is hard-coded, probably not a big deal however
            shader->setEntryPoint("main");

            shader->setOverrideVersion(460);

            // put things like #define #undef or other code to be processed
            // right after the version declaration
            // shader->setPreamble(PreambleString.c_str());

            // processes is a vector of strings that stores descriptions on the
            // processes performed upon the shader like define-macro undef-macro preamble-text etc
            // those 3 are the only things appended to this in the standalone code
            // shader->addProcesses(processes);

            // The following can be done to shift binding and set indices for resources
            // for (int r = 0; r < glslang::EResCount; ++r)
            // {
            //     glslang::TResourceType const res = glslang::TResourceType(r);
            //
            //     // Set base bindings
            //     shader->setShiftBinding(res, baseBinding[res][compUnit.stage]);
            //
            //     // Set bindings for particular resource sets
            //     // TODO: use a range based for loop here, when available in all environments.
            //     for (auto i = baseBindingForSet[res][compUnit.stage].begin();
            //          i != baseBindingForSet[res][compUnit.stage].end();
            //          ++i)
            //         shader->setShiftBindingForSet(res, i->second, i->first);
            // }

            // do the following to use unknown image format
            // shader->setNoStorageFormat(true / false);

            // do the following to set resource set binding for current stage
            // shader->setResourceSetBinding(baseResourceSetBinding[compUnit.stage]);

            shader->setEnhancedMsgs();

            // shader->setDebugInfo(true);

            // Set up the environment, some subsettings take precedence over earlier
            // ways of setting things.
            shader->setEnvInput(
                glslang::EShSourceGlsl, compUnit.stage, glslang::EShClient::EShClientVulkan, 100);

            shader->setEnvClient(glslang::EShClient::EShClientVulkan,
                                 glslang::EShTargetClientVersion::EShTargetVulkan_1_3);

            shader->setEnvTarget(glslang::EShTargetLanguage::EShTargetSpv,
                                 glslang::EShTargetLanguageVersion::EShTargetSpv_1_6);

            shaders.push_back(shader);

            TBuiltInResource resource {
                .maxDrawBuffers = 1,
                .limits         = { .generalUniformIndexing = true, .generalSamplerIndexing = true }
            };

            if (!shader->parse(&resource, 100, false, messages, includer))
            {
                compileFailed = true;
            };

            program.addShader(shader);

            logger::info("[SHADER COMPILATION] {}:\n{}\n{}",
                         compUnit.fileName[0].c_str(),
                         shader->getInfoLog(),
                         shader->getInfoDebugLog());
        }

        //
        // Program-level processing...
        //

        // Map IO
        // if (!program.mapIO())
        // {
        //     linkFailed = true;
        // };

        // Report
        // logger::info("[SHADER PROGRAM] {}\n{}", program.getInfoLog(), program.getInfoDebugLog());

        // Reflect
        // ************************************ TODO(aether) What is this? ********************************************
        // program.buildReflection(ReflectOptions);
        // program.dumpReflection();

        std::vector<std::string> outputFiles;

        // Dump SPIR-V
        compileOrLinkFailed.fetch_or(compileFailed);
        compileOrLinkFailed.fetch_or(linkFailed);

        if (static_cast<bool>(compileOrLinkFailed.load()))
            logger::error("SPIR-V is not generated for failed compile or link");
        else
        {
            std::vector<glslang::TIntermediate*> intermediates;

            for (int stage = 0; stage < EShLangCount; ++stage)
            {
                if (auto* i = program.getIntermediate(static_cast<EShLanguage>(stage)))
                {
                    intermediates.emplace_back(i);
                }
            }

            for (auto [i, intermediate] : std::ranges::views::enumerate(intermediates))
            {
                std::vector<uint32_t> spirv;
                spv::SpvBuildLogger logger;
                glslang::SpvOptions spvOptions;

                // if (Options & EOptionDebug)
                // {
                //     spvOptions.generateDebugInfo = true;
                //     if (emitNonSemanticShaderDebugInfo)
                //     {
                //         spvOptions.emitNonSemanticShaderDebugInfo = true;
                //         if (emitNonSemanticShaderDebugSource)
                //         {
                //             spvOptions.emitNonSemanticShaderDebugSource = true;
                //         }
                //     }
                // }
                // else if (stripDebugInfo)
                //     spvOptions.stripDebugInfo = true;

                spvOptions.disableOptimizer = false;
                spvOptions.optimizeSize     = false;
                spvOptions.disassemble      = false;
                spvOptions.validate         = true;
                spvOptions.compileOnly      = false;

                glslang::GlslangToSpv(*intermediate, spirv, &logger, &spvOptions);

                logger::info("{}", logger.getAllMessages().c_str());

                std::string filename = std::format("{}.spv", i);

                MC_ASSERT(glslang::OutputSpvBin(spirv, filename.c_str()));

                outputFiles.push_back(filename);

                spv::Disassemble(std::cout, spirv);
            }
        }

        compileOrLinkFailed.fetch_or(compileFailed);
        compileOrLinkFailed.fetch_or(linkFailed);

        if (!static_cast<bool>(compileOrLinkFailed.load()))
        {
            std::set<std::string> includedFiles = includer.getIncludedFiles();
            sources.insert(sources.end(), includedFiles.begin(), includedFiles.end());

            logger::warn("Wrote a dependency_file.txt, check it!");
            writeDepFile("./dependency_file.txt", outputFiles, sources);
        }

        while (shaders.size() > 0)
        {
            delete shaders.back();
            shaders.pop_back();
        }
    }

    void CompileAndLinkShaderFiles(TWorklist& Worklist)
    {
        std::vector<ShaderCompUnit> compUnits;

        // Transfer all the work items from to a simple list of
        // of compilation units.  (We don't care about the thread
        // work-item distribution properties in this path, which
        // is okay due to the limited number of shaders, know since
        // they are all getting linked together)
        TWorkItem* workItem;

        while (Worklist.remove(workItem))
        {
            EShLanguage const stage = std::unordered_map<std::string_view, EShLanguage> {
                { ".vert",  EShLangVertex         },
                { ".tesc",  EShLangTessControl    },
                { ".tese",  EShLangTessEvaluation },
                { ".geom",  EShLangGeometry       },
                { ".frag",  EShLangFragment       },
                { ".comp",  EShLangCompute        },
                { ".rgen",  EShLangRayGen         },
                { ".rint",  EShLangIntersect      },
                { ".rahit", EShLangAnyHit         },
                { ".rchit", EShLangClosestHit     },
                { ".rmiss", EShLangMiss           },
                { ".rcall", EShLangCallable       },
                { ".mesh",  EShLangMesh           },
                { ".task",  EShLangTask           },
            }[std::filesystem::path(workItem->name).extension().string()];

            ShaderCompUnit compUnit(stage);

            std::ifstream file(workItem->name.c_str(), std::ios::ate | std::ios::binary);

            MC_ASSERT_MSG(file.is_open(), "Failed to read file '{}'", workItem->name.c_str());

            size_t fileSize = file.tellg();

            char* glslCode = new char[fileSize];

            file.seekg(0);
            file.read(glslCode, fileSize);

            file.close();

            compUnit.addString(workItem->name, glslCode);
            compUnits.push_back(compUnit);
        }

        CompileAndLinkShaderUnits(compUnits);

        // free memory from ReadFileData, which got stored in a const char*
        // as the first string above
        for (auto it = compUnits.begin(); it != compUnits.end(); ++it)
            delete[] (const_cast<char*>(it->text[0]));
    }
};  // namespace

namespace renderer::backend
{
    Shader::Shader(std::filesystem::path path) : Shader {}
    {
        TWorklist workList;
        std::vector<TWorkItem> workItems { TWorkItem("../../shaders/fs.frag") };

        std::for_each(workItems.begin(),
                      workItems.end(),
                      [&workList](TWorkItem& item)
                      {
                          workList.add(&item);
                      });

        glslang::InitializeProcess();
        CompileAndLinkShaderFiles(workList);
        glslang::FinalizeProcess();

        MC_ASSERT(!(compileFailed.load() || linkFailed.load()));
    };
}  // namespace renderer::backend
