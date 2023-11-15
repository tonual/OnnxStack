using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OnnxStack.Console.Runner;

namespace OnnxStack.Console
{
    internal class AppService : IHostedService
    {
        private IExampleRunner selectedRunner;
        public AppService(IExampleRunner exampleRunner)
        {
            selectedRunner = exampleRunner;
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
                
            
                await selectedRunner.RunAsync();
            
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            return Task.CompletedTask;
        }
    }
}