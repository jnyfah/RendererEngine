﻿using Panzerfaust.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Panzerfaust.Service
{
    internal class ProjectService : IProjectService
    {
        private readonly string _cachePath = Path.Combine(".", "Cache");
        private readonly string _fileExtension = ".pzf";

        private void EnsureCacheDirectoryIsPresent()
        {
            if (!Directory.Exists(_cachePath))
            {
                Directory.CreateDirectory(_cachePath);
            }
        }

        public async Task<Project?> CreateAsync(string name, string path, DateTime creationTime, DateTime updateTime)
        {
            EnsureCacheDirectoryIsPresent();
            
            Project p = new() { Name = name, Fullpath = path, CreationDate = creationTime, UpdateDate = updateTime };

            using var fs = File.OpenWrite(Path.Combine(_cachePath, $"{p.GetHash()}{_fileExtension}"));
            await JsonSerializer.SerializeAsync(fs, p).ConfigureAwait(false);
            return p;
        }

        public async Task<IEnumerable<Project>> LoadProjectsAsync()
        {
            EnsureCacheDirectoryIsPresent();
            
            var output = new List<Project>();

            foreach (var file in Directory.EnumerateFiles(_cachePath))
            {
                if (new DirectoryInfo(file).Extension != _fileExtension) continue;

                using var fs = File.OpenRead(file);
                var project = await JsonSerializer.DeserializeAsync<Project>(fs).ConfigureAwait(false);

                if (project == null) continue;

                output.Add(project);
            }

            return output;
        }

        public async Task SaveAsync(Project p)
        {
            EnsureCacheDirectoryIsPresent();

            using var fs = File.OpenWrite(Path.Combine(_cachePath, $"{p.GetHash()}{_fileExtension}"));
            await JsonSerializer.SerializeAsync(fs, p).ConfigureAwait(false);
        }

        public Task<(bool, ErrorMessage?)> DeleteAsync(Project p)
        {
            return Task.Run(() =>
            {
                (bool, ErrorMessage?) result = (false, null);

                try
                {
                    EnsureCacheDirectoryIsPresent();

                    var pzffilepath = Path.Combine(_cachePath, $"{p.GetHash()}{_fileExtension}");
                    File.Delete(pzffilepath);

                    Directory.Delete(p.Fullpath, true);

                    result = (true, null);
                }
                catch (Exception e)
                {
                    result = (false, new() { Message = e.Message, Exception = e });
                }

                return result;
            });
        }
    }
}
