﻿using Microsoft.ML.OnnxRuntime;
using OnnxStack.Common.Config;
using System;
using System.Text.Json.Serialization;

namespace OnnxStack.Core.Config
{
    public class OnnxStackConfig : IConfigSection
    {
        /// <summary>
        /// Gets or sets the device identifier.
        /// </summary>
        /// <value>
        /// The device identifier used by DirectML and CUDA.
        /// </value>
        public int DeviceId { get; set; }

        /// <summary>
        /// Gets or sets the execution provider target.
        /// </summary>
        public ExecutionProvider ExecutionProviderTarget { get; set; } = ExecutionProvider.DirectML;

        public string OnnxTokenizerPath { get; set; }
        public string OnnxUnetPath { get; set; }
        public string OnnxVaeDecoderPath { get; set; }
        public string OnnxVaeEncoderPath { get; set; }
        public string OnnxTextEncoderPath { get; set; }
        public string OnnxSafetyModelPath { get; set; }
        public bool IsSafetyModelEnabled { get; set; }
        public ExecutionMode ExecutionMode { get; set; }
        public int InterOpNumThreads { get; set; }
        public int IntraOpNumThreads { get; set; }

        public void Initialize()
        {
            IsSafetyModelEnabled = false;
        }
    }
}
