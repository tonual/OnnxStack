using MathNet.Numerics;
using OnnxStack.StableDiffusion;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Models;
using SixLabors.ImageSharp;

namespace OnnxStack.Console.Runner
{
    public sealed class RafStableDiffusion : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly string _inputDirectory;
        private readonly IStableDiffusionService _stableDiffusionService;

        public RafStableDiffusion(IStableDiffusionService stableDiffusionService)
        {
            _stableDiffusionService = stableDiffusionService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(RafStableDiffusion));
            _inputDirectory = Path.Combine(_outputDirectory, "InputImage");
        }

        public string Name => "Raf SD fixture";
        public string Description => "creates a series of diffusions from image input";
        
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);
            
            var prompt = "Transdimensional travel, multiverse, transcendent travel";           
            var negativePrompt = "cartoon, low contrast, noisy, low quality";
            var inputImg = "IMG_4198.jpg";
        
            var promptOptions = new PromptOptions
            {
                Prompt = prompt,
                NegativePrompt = negativePrompt,
                DiffuserType = StableDiffusion.Enums.DiffuserType.ImageToImage,
                InputImage = new InputImage(
                    File.ReadAllBytes(Path.Combine(_inputDirectory,inputImg))
                )
            };

            var schedulerOptions = new SchedulerOptions
            {
                SchedulerType = StableDiffusion.Enums.SchedulerType.Euler,
                Seed = 438233955, //Random.Shared.Next()
                //SchedulerType = 
                InferenceSteps = 15,
                GuidanceScale = 6,
                Strength = 0.5f
            };
            
            var x = _stableDiffusionService.Models;
            
            foreach (var model in _stableDiffusionService.Models)
            {
                OutputHelpers.WriteConsole($"Loading Model `{model.Name}`...", ConsoleColor.Green);
                await _stableDiffusionService.LoadModel(model);                                    
                await GenerateImage(model, promptOptions, schedulerOptions);                
                OutputHelpers.WriteConsole($"Unloading Model `{model.Name}`...", ConsoleColor.Green);
                await _stableDiffusionService.UnloadModel(model);
            }
        }

        private async Task<bool> GenerateImage(ModelOptions model, PromptOptions prompt, SchedulerOptions options)
        {
            var outputFilename = Path.Combine(_outputDirectory, $"{DateTime.Now.Ticks}_{options.SchedulerType}.png");
            var result = await _stableDiffusionService.GenerateAsImageAsync(model, prompt, options);
            if (result == null)
                return false;

            await result.SaveAsPngAsync(outputFilename);
            OutputHelpers.WriteConsole($"{options.SchedulerType} Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
            return true;
        }
    }
}
