﻿using MathNet.Numerics;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers
{
    public sealed class LMSScheduler : SchedulerBase
    {
        private float[] _sigmas;
        private readonly List<DenseTensor<float>> _derivatives;

        /// <summary>
        /// Initializes a new instance of the <see cref="LMSScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public LMSScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMSScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public LMSScheduler(SchedulerOptions schedulerOptions) : base(schedulerOptions)
        {
            _derivatives = new List<DenseTensor<float>>();
        }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            var betas = Enumerable.Empty<float>();
            if (!Options.TrainedBetas.IsNullOrEmpty())
            {
                betas = Options.TrainedBetas;
            }
            else if (Options.BetaSchedule == BetaScheduleType.Linear)
            {
                var steps = Options.TrainTimesteps - 1;
                var delta = Options.BetaStart + (Options.BetaEnd - Options.BetaStart);
                betas = Enumerable.Range(0, Options.TrainTimesteps)
                    .Select(i => delta * i / steps);
            }
            else if (Options.BetaSchedule == BetaScheduleType.ScaledLinear)
            {
                var start = (float)Math.Sqrt(Options.BetaStart);
                var end = (float)Math.Sqrt(Options.BetaEnd);
                betas = np.linspace(start, end, Options.TrainTimesteps)
                    .ToArray<float>()
                    .Select(x => x * x);
            }
            else if (Options.BetaSchedule == BetaScheduleType.SquaredCosCapV2)
            {
                betas = GetBetasForAlphaBar();
            }


            var alphas = betas.Select(beta => 1 - beta);
            var cumulativeProduct = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b));

            // Create sigmas as a list and reverse it
            _sigmas = cumulativeProduct
                .Select(alpha_prod => (float)Math.Sqrt((1 - alpha_prod) / alpha_prod))
                .ToArray();

            // standard deviation of the initial noise distrubution
            var maxSigma = _sigmas.Max();
            var initNoiseSigma = Options.TimestepSpacing == TimestepSpacingType.Linspace || Options.TimestepSpacing == TimestepSpacingType.Trailing
                ? maxSigma
                : (float)Math.Sqrt(maxSigma * maxSigma + 1);
            SetInitNoiseSigma(initNoiseSigma);
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        { 
            NDArray timestepsArray = null;
            if (Options.TimestepSpacing == TimestepSpacingType.Linspace)
            {
                timestepsArray = np.linspace(0, Options.TrainTimesteps - 1, Options.InferenceSteps);
                timestepsArray = np.around(timestepsArray);
            }
            else if (Options.TimestepSpacing == TimestepSpacingType.Leading)
            {
                var stepRatio = Options.TrainTimesteps / Options.InferenceSteps;
                timestepsArray = np.arange(0, (float)Options.InferenceSteps) * stepRatio;
                timestepsArray = np.around(timestepsArray);
                timestepsArray += Options.StepsOffset;
            }
            else if (Options.TimestepSpacing == TimestepSpacingType.Trailing)
            {
                var stepRatio = Options.TrainTimesteps / (Options.InferenceSteps - 1);
                timestepsArray = np.arange(Options.TrainTimesteps, 0f, -stepRatio)["::-1"];
                timestepsArray = np.around(timestepsArray);
                timestepsArray -= 1;
            }

            var sigmas = _sigmas.ToArray();
            var timesteps = timestepsArray.ToArray<float>();
            var log_sigmas = np.log(sigmas).ToArray<float>();
            var range = np.arange(0, (float)_sigmas.Length).ToArray<float>();
            sigmas = Interpolate(timesteps, range, _sigmas);

            if (Options.UseKarrasSigmas)
            {
                sigmas = ConvertToKarras(sigmas);
                timesteps = SigmaToTimestep(sigmas, log_sigmas);
            }

            //  add 0.000 to the end of the sigmas
            _sigmas = sigmas.Append(0.000f).ToArray();

            return timesteps.Select(x => (int)x)
                 .OrderByDescending(x => x)
                 .ToArray();
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="sample">The sample.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public override DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            // Get step index of timestep from TimeSteps
            int stepIndex = Timesteps.IndexOf(timestep);

            // Get sigma at stepIndex
            var sigma = _sigmas[stepIndex];
            sigma = (float)Math.Sqrt(Math.Pow(sigma, 2) + 1);

            // Divide sample tensor shape {2,4,(H/8),(W/8)} by sigma
            sample = sample.DivideTensorByFloat(sigma, sample.Dimensions);

            return sample;
        }


        /// <summary>
        /// Processes a inference step for the specified model output.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="order">The order.</param>
        /// <returns></returns>
        public override DenseTensor<float> Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            int stepIndex = Timesteps.IndexOf(timestep);
            var sigma = _sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            // sample.SubtractTensors(modelOutput.MultipleTensorByFloat(sigma));

            // 2. Convert to an ODE derivative
            var derivativeSample = sample
                .SubtractTensors(sample.SubtractTensors(modelOutput.MultipleTensorByFloat(sigma)))
                .DivideTensorByFloat(sigma, sample.Dimensions);

            _derivatives.Add(derivativeSample);
            if (_derivatives.Count > order)
            {
                // remove first element
                _derivatives.RemoveAt(0);
            }

            // 3. compute linear multistep coefficients
            order = Math.Min(stepIndex + 1, order);
            var lmsCoeffs = Enumerable.Range(0, order)
                .Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder));

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives
            var revDerivatives = Enumerable.Reverse(_derivatives);

            // Create list of tuples from the lmsCoeffs and reversed derivatives
            var lmsCoeffsAndDerivatives = lmsCoeffs
                .Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative))
                .ToArray();

            // Create tensor for product of lmscoeffs and derivatives
            var lmsDerProduct = new DenseTensor<float>[_derivatives.Count];
            for (int i = 0; i < lmsCoeffsAndDerivatives.Length; i++)
            {
                // Multiply to coeff by each derivatives to create the new tensors
                var (lmsCoeff, derivative) = lmsCoeffsAndDerivatives[i];
                lmsDerProduct[i] = derivative.MultipleTensorByFloat((float)lmsCoeff);
            }

            // Add the sumed tensor to the sample
            return sample.AddTensors(lmsDerProduct.SumTensors(modelOutput.Dimensions));
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="originalSamples">The original samples.</param>
        /// <param name="noise">The noise.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        public override DenseTensor<float> AddNoise(DenseTensor<float> originalSamples, DenseTensor<float> noise, IReadOnlyList<int> timesteps)
        {
            // Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py#L439
            var sigma = timesteps
                .Select(x => Timesteps.IndexOf(x))
                .Select(x => _sigmas[x])
                .Max();

            return noise
                .MultipleTensorByFloat(sigma)
                .AddTensors(originalSamples);
        }


        /// <summary>
        /// Gets the LMS coefficient.
        /// </summary>
        /// <param name="order">The order.</param>
        /// <param name="t">The t.</param>
        /// <param name="currentOrder">The current order.</param>
        /// <returns></returns>
        private double GetLmsCoefficient(int order, int t, int currentOrder)
        {
            // python line 135 of scheduling_lms_discrete.py
            // Compute a linear multistep coefficient.
            double LmsDerivative(double tau)
            {
                double prod = 1.0;
                for (int k = 0; k < order; k++)
                {
                    if (currentOrder == k)
                    {
                        continue;
                    }
                    prod *= (tau - _sigmas[t - k]) / (_sigmas[t - currentOrder] - _sigmas[t - k]);
                }
                return prod;
            }
            return Integrate.OnClosedInterval(LmsDerivative, _sigmas[t], _sigmas[t + 1], 1e-4);
        }

        protected override void Dispose(bool disposing)
        {
            _sigmas = null;
            _derivatives?.Clear();
            base.Dispose(disposing);
        }
    }
}
