﻿using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Diffusers.LatentConsistency;
using OnnxStack.StableDiffusion.Enums;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Pipelines
{
    public sealed class LatentConsistencyPipeline : IPipeline
    {
        private readonly DiffuserPipelineType _pipelineType;
        private readonly ILogger<LatentConsistencyPipeline> _logger;
        private readonly ConcurrentDictionary<DiffuserType, IDiffuser> _diffusers;

        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConsistencyPipeline"/> class.
        /// </summary>
        /// <param name="onnxModelService">The onnx model service.</param>
        /// <param name="promptService">The prompt service.</param>
        public LatentConsistencyPipeline(IEnumerable<IDiffuser> diffusers, ILogger<LatentConsistencyPipeline> logger)
        {
            _logger = logger;
            _pipelineType = DiffuserPipelineType.LatentConsistency;
            _diffusers = diffusers
                .Where(x => x.PipelineType == _pipelineType)
                .ToConcurrentDictionary(k => k.DiffuserType, v => v);
        }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public DiffuserPipelineType PipelineType => _pipelineType;


        /// <summary>
        /// Gets the diffusers.
        /// </summary>
        public ConcurrentDictionary<DiffuserType, IDiffuser> Diffusers => _diffusers;


        /// <summary>
        /// Gets the diffuser.
        /// </summary>
        /// <param name="diffuserType">Type of the diffuser.</param>
        /// <returns></returns>
        public IDiffuser GetDiffuser(DiffuserType diffuserType)
        {
            _diffusers.TryGetValue(diffuserType, out var diffuser);
            return diffuser;
        }
    }
}
