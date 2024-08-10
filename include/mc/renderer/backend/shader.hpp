#pragma once

#include <filesystem>
#include <fstream>
#include <list>
#include <mutex>

#include <glslang/Public/ShaderLang.h>
#include <set>

namespace renderer::backend
{
    class DirStackFileIncluder : public glslang::TShader::Includer
    {
    public:
        DirStackFileIncluder() : m_externalLocalDirectoryCount(0) {}

        auto includeLocal(char const* headerName,
                          char const* includerName,
                          size_t inclusionDepth) -> IncludeResult* override
        {
            return readLocalPath(headerName, includerName, (uint32_t)inclusionDepth);
        }

        auto includeSystem(char const* headerName,
                           char const* /*includerName*/,
                           size_t /*inclusionDepth*/) -> IncludeResult* override
        {
            return readSystemPath(headerName);
        }

        // Externally set directories. E.g., from a command-line -I<dir>.
        //  - Most-recently pushed are checked first.
        //  - All these are checked after the parse-time stack of local directories
        //    is checked.
        //  - This only applies to the "local" form of #include.
        //  - Makes its own copy of the path.
        void pushExternalLocalDirectory(std::string const& dir)
        {
            m_directoryStack.push_back(dir);
            m_externalLocalDirectoryCount = (int)m_directoryStack.size();
        }

        void releaseInclude(IncludeResult* result) override
        {
            if (result != nullptr)
            {
                delete[] static_cast<char*>(result->userData);
                delete result;
            }
        }

        std::set<std::string> getIncludedFiles() { return m_includedFiles; }

        ~DirStackFileIncluder() override {}

    protected:
        std::vector<std::string> m_directoryStack;
        int m_externalLocalDirectoryCount;
        std::set<std::string> m_includedFiles;

        // Search for a valid "local" path based on combining the stack of include
        // directories and the nominal name of the header.
        virtual IncludeResult* readLocalPath(char const* headerName, char const* includerName, uint32_t depth)
        {
            // Discard popped include directories, and
            // initialize when at parse-time first level.
            m_directoryStack.resize(depth + static_cast<uint32_t>(m_externalLocalDirectoryCount));
            if (depth == 1)
                m_directoryStack.back() = getDirectory(includerName);

            // Find a directory that works, using a reverse search of the include stack.
            for (auto it = m_directoryStack.rbegin(); it != m_directoryStack.rend(); ++it)
            {
                std::string path = *it + '/' + headerName;
                std::replace(path.begin(), path.end(), '\\', '/');
                std::ifstream file(path, std::ios_base::binary | std::ios_base::ate);
                if (file)
                {
                    m_directoryStack.push_back(getDirectory(path));
                    m_includedFiles.insert(path);
                    return newIncludeResult(path, file, static_cast<size_t>(file.tellg()));
                }
            }

            return nullptr;
        }

        // Search for a valid <system> path.
        // Not implemented yet; returning nullptr signals failure to find.
        virtual IncludeResult* readSystemPath(char const* /*headerName*/) const { return nullptr; }

        // Do actual reading of the file, filling in a new include result.
        virtual IncludeResult*
        newIncludeResult(std::string const& path, std::ifstream& file, size_t length) const
        {
            char* content = new char[length];
            file.seekg(0, file.beg);
            file.read(content, static_cast<int>(length));
            return new IncludeResult(path, content, length, content);
        }

        // If no path markers, return current working directory.
        // Otherwise, strip file name and return path leading up to it.
        virtual std::string getDirectory(std::string const path) const
        {
            size_t last = path.find_last_of("/\\");
            return last == std::string::npos ? "." : path.substr(0, last);
        }
    };

    class TWorkItem
    {
    public:
        TWorkItem() = default;

        explicit TWorkItem(std::string const& s) : name(s) {}

        std::string name;
        std::string results;
        std::string resultsIndex;
    };

    class TWorklist
    {
    public:
        TWorklist()  = default;
        ~TWorklist() = default;

        void add(TWorkItem* item)
        {
            std::lock_guard<std::mutex> guard(mutex);

            worklist.push_back(item);
        }

        bool remove(TWorkItem*& item)
        {
            std::lock_guard<std::mutex> guard(mutex);

            if (worklist.empty())
                return false;

            item = worklist.front();

            worklist.pop_front();

            return true;
        }

        auto size() -> size_t { return worklist.size(); }

        bool empty() { return worklist.empty(); }

    protected:
        std::mutex mutex;

        std::list<TWorkItem*> worklist;
    };

    class Shader
    {
    public:
        Shader()  = default;
        ~Shader() = default;

        Shader(std::filesystem::path path);

        Shader(Shader&&)                    = default;
        auto operator=(Shader&&) -> Shader& = default;

        Shader(Shader const&)                    = delete;
        auto operator=(Shader const&) -> Shader& = delete;

        auto getSpirv() const -> std::vector<uint32_t> const& { return m_spirv; }

    private:
        std::vector<uint32_t> m_spirv {};
    };
}  // namespace renderer::backend
