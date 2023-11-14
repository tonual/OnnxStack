﻿using LibGit2Sharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.UI.Helpers
{
    public class RepositoryDownloader : IDisposable
    {
        private readonly LFSFilter _lfsFilter;
        private readonly string _repositoryUrl;
        private readonly string _destinationPath;
        private readonly CloneOptions _cloneOptions;
        private readonly Action<string, double> _progressCallback;
        private readonly CancellationTokenSource _cancellationTokenSource;

        public RepositoryDownloader(string repositoryUrl, string destinationPath, Action<string, double> progressCallback = null, CancellationToken cancellationToken = default)
        {
            _repositoryUrl = repositoryUrl;
            _destinationPath = destinationPath;
            _progressCallback = progressCallback;
            _cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            _lfsFilter = new LFSFilter("lfs", new[] { new FilterAttributeEntry("lfs") });
            _cloneOptions = new CloneOptions { OnCheckoutProgress = (f, p, t) => _progressCallback?.Invoke(f, (p * 100.0 / t)) };
            if (!GlobalSettings.GetRegisteredFilters().Any(x => x.Name == "lfs"))
                GlobalSettings.RegisterFilter(_lfsFilter);
        }


        public Task DownloadAsync()
        {
            if (!Directory.Exists(_destinationPath))
                Directory.CreateDirectory(_destinationPath);

            try
            {
                // Perform the clone operation.
                return Task.Factory.StartNew(() =>
                    Repository.Clone(_repositoryUrl, _destinationPath, _cloneOptions),
                    _cancellationTokenSource.Token,
                    TaskCreationOptions.LongRunning,
                    TaskScheduler.Default);
            }
            catch (Exception)
            {
                _lfsFilter.KillProcess();
                throw;
            }
        }

        public void Cancel()
        {
            _cancellationTokenSource.Cancel();
            _lfsFilter.KillProcess();
        }

        public void Dispose()
        {
            Cancel();
        }
    }

    public class LFSFilter : Filter
    {
        private Process _process;
        private FilterMode _mode;


        /// <summary>
        /// Initializes a new instance of the <see cref="LFSFilter"/> class.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="attributes"></param>
        public LFSFilter(string name, IEnumerable<FilterAttributeEntry> attributes)
            : base(name, attributes)
        {
            // Kill the downloads if the app exists or crashes
            AppDomain.CurrentDomain.DomainUnload += (s, e) => KillProcess();
            AppDomain.CurrentDomain.ProcessExit += (s, e) => KillProcess();
            AppDomain.CurrentDomain.UnhandledException += (s, e) => KillProcess();
        }


        /// <summary>
        /// Kills the process.
        /// </summary>
        public void KillProcess()
        {
            try
            {
                if (_process is not null)
                {
                    _process.Kill(true);
                    _process.WaitForExit();
                }
            }
            catch (Exception)
            {

            }
        }


        /// <summary>
        /// Initialize callback on filter
        /// </summary>
        protected override void Initialize()
        {
            base.Initialize();
        }


        /// <summary>
        /// Clean the input stream and write to the output stream.
        /// </summary>
        /// <param name="path">The path of the file being filtered</param>
        /// <param name="root">The path of the working directory for the owning repository</param>
        /// <param name="input">Input from the upstream filter or input reader</param>
        /// <param name="output">Output to the downstream filter or output writer</param>
        protected override void Clean(string path, string root, Stream input, Stream output)
        {
            try
            {
                // write file data to stdin
                input.CopyTo(_process.StandardInput.BaseStream);
                input.Flush();
            }
            catch (Exception)
            {

            }
        }


        /// <summary>
        /// Complete callback on filter
        /// This optional callback will be invoked when the upstream filter is
        /// closed. Gives the filter a chance to perform any final actions or
        /// necissary clean up.
        /// </summary>
        /// <param name="path">The path of the file being filtered</param>
        /// <param name="root">The path of the working directory for the owning repository</param>
        /// <param name="output">Output to the downstream filter or output writer</param>
        protected override void Complete(string path, string root, Stream output)
        {
            try
            {
                // finalize stdin and wait for git-lfs to finish
                _process.StandardInput.Flush();
                _process.StandardInput.Close();
                if (_mode == FilterMode.Clean)
                {
                    _process.WaitForExit();

                    // write git-lfs pointer for 'clean' to git or file data for 'smudge' to working copy
                    _process.StandardOutput.BaseStream.CopyTo(output);
                    _process.StandardOutput.BaseStream.Flush();
                    _process.StandardOutput.Close();
                    output.Flush();
                    output.Close();
                }
                else if (_mode == FilterMode.Smudge)
                {
                    // write git-lfs pointer for 'clean' to git or file data for 'smudge' to working copy
                    _process.StandardOutput.BaseStream.CopyTo(output);
                    _process.StandardOutput.BaseStream.Flush();
                    _process.StandardOutput.Close();
                    output.Flush();
                    output.Close();

                    _process.WaitForExit();
                }

                _process.Dispose();
            }
            catch (Exception)
            {

            }
        }


        /// <summary>
        /// Indicates that a filter is going to be applied for the given file for
        /// the given mode.
        /// </summary>
        /// <param name="path">The path of the file being filtered</param>
        /// <param name="root">The path of the working directory for the owning repository</param>
        /// <param name="mode">The filter mode</param>
        protected override void Create(string path, string root, FilterMode mode)
        {
            try
            {
                _mode = mode;
                // launch git-lfs
                _process = new Process();
                _process.StartInfo.FileName = "git-lfs";
                _process.StartInfo.Arguments = string.Format("{0} {1}", mode == FilterMode.Clean ? "clean" : "smudge", path);
                _process.StartInfo.WorkingDirectory = root;
                _process.StartInfo.RedirectStandardInput = true;
                _process.StartInfo.RedirectStandardOutput = true;
                _process.StartInfo.RedirectStandardError = true;
                _process.StartInfo.CreateNoWindow = true;
                _process.StartInfo.UseShellExecute = false;
                _process.Start();
            }
            catch
            {
                throw new Exception("Git-LFS is not installed or is not initialized");
            }
        }


        /// <summary>
        /// Smudge the input stream and write to the output stream.
        /// </summary>
        /// <param name="path">The path of the file being filtered</param>
        /// <param name="root">The path of the working directory for the owning repository</param>
        /// <param name="input">Input from the upstream filter or input reader</param>
        /// <param name="output">Output to the downstream filter or output writer</param>
        protected override void Smudge(string path, string root, Stream input, Stream output)
        {
            try
            {
                // write git-lfs pointer to stdin
                input.CopyTo(_process.StandardInput.BaseStream);
                input.Flush();
            }
            catch (Exception)
            {

            }
        }
    }
}
