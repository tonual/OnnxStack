using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OnnxStack.Console.Runner;
using OnnxStack.Core;
using System.Reflection;

namespace OnnxStack.Console
{
    internal class Program
    {
        static async Task Main(string[] _)
        {
            var builder = Host.CreateApplicationBuilder();
            builder.Logging.ClearProviders();
            builder.Services.AddLogging((loggingBuilder) => loggingBuilder.SetMinimumLevel(LogLevel.Error));

            // Add OnnxStack
            builder.Services.AddOnnxStack();
            builder.Services.AddOnnxStackStableDiffusion();

            // Add AppService
            builder.Services.AddHostedService<AppService>();
            
            builder.Services.AddSingleton<IExampleRunner, RafStableDiffusion>();
            // Start
            await builder.Build().RunAsync();
        }
    }
}