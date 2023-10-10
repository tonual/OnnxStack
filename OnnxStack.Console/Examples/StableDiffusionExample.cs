﻿using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using SixLabors.ImageSharp;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly IStableDiffusionService _stableDiffusionService;

        public StableDiffusionExample(IStableDiffusionService stableDiffusionService)
        {
            _stableDiffusionService = stableDiffusionService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionExample));
        }

        public string Name => "Stable Diffusion Demo";

        public string Description => "Creates images from the text prompt provided using all Scheduler types";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);

            while (true)
            {
                OutputHelpers.WriteConsole("Please type a prompt and press ENTER", ConsoleColor.Yellow);
                var prompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                OutputHelpers.WriteConsole("Please type a negative prompt and press ENTER (optional)", ConsoleColor.Yellow);
                var negativePrompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                var promptOptions = new PromptOptions
                {
                    Prompt = prompt,
                    NegativePrompt = negativePrompt,
                   
                };

                var schedulerOptions = new SchedulerOptions
                {
                    Seed = Random.Shared.Next()
                };
                foreach (var schedulerType in Enum.GetValues<SchedulerType>())
                {
                    promptOptions.SchedulerType = schedulerType;

                    OutputHelpers.WriteConsole("Generating Image...", ConsoleColor.Green);
                    await GenerateImage(promptOptions, schedulerOptions);
                }
            }
        }

        private async Task<bool> GenerateImage(PromptOptions prompt, SchedulerOptions options)
        {
            var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{prompt.SchedulerType}.png");
            var result = await _stableDiffusionService.GenerateAsImageAsync(prompt, options);
            if (result == null)
                return false;

            await result.SaveAsPngAsync(outputFilename);
            OutputHelpers.WriteConsole($"{prompt.SchedulerType} Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
            return true;
        }
    }
}
