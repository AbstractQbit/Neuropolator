using System;
using System.Diagnostics;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenTabletDriver.Plugin;
using OpenTabletDriver.Plugin.Attributes;
using OpenTabletDriver.Plugin.Logging;
using OpenTabletDriver.Plugin.Output;
using OpenTabletDriver.Plugin.Tablet;
using OpenTabletDriver.Plugin.Timing;

namespace Neuropolator;

public class TestFilter : IPositionedPipelineElement<IDeviceReport>//, IDisposable
{
    public TestFilter() : base()
    {
        Log.Debug("Neuropolator", inputName);
        Log.Debug("Neuropolator", outputName);
        Log.Debug("Neuropolator", ort_session.InputMetadata[inputName].Dimensions.ToString());
    }
    public PipelinePosition Position => PipelinePosition.Pixels;

    public event Action<IDeviceReport>? Emit;

    private static InferenceSession ort_session = new InferenceSession("./aaaaa1.onnx", new SessionOptions
    // private static InferenceSession ort_session = new InferenceSession("./dilated3_ar5steps.onnx", new SessionOptions
    {
        InterOpNumThreads = 1,
        IntraOpNumThreads = 1,
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
    });
    private static string inputName = ort_session.InputMetadata.Keys.First();
    private static string outputName = ort_session.OutputMetadata.Keys.First();

    private static readonly int[] in_shape = ort_session.InputMetadata[inputName].Dimensions;
    private static readonly long[] in_shape_long = in_shape.Select(x => (long)x).ToArray();
    private static int in_ctx = in_shape[2];
    private static int out_steps = ort_session.OutputMetadata[outputName].Dimensions[2];
    private static readonly RunOptions runOptions = new RunOptions();

    private DenseTensor<float> DeltasHist = new(in_shape, reverseStride: false);
    private Vector2 previousPosition = Vector2.Zero;
    private HPETDeltaStopwatch reportStopwatch = new HPETDeltaStopwatch();

    private float scale = 0.3f;

    private int stepsToTake = 2;



    public void Consume(IDeviceReport value)
    {
        if (value is IAbsolutePositionReport report)
        {
            // Reset logic
            if (reportStopwatch.Restart().TotalMilliseconds > 50)
            {
                DeltasHist = new DenseTensor<float>(in_shape, reverseStride: false);
                previousPosition = report.Position;
            }

            var delta = (report.Position - previousPosition) * scale;
            previousPosition = report.Position;

            // Roll tensor along L
            foreach (int l in Enumerable.Range(0, in_ctx - 1))
            {
                DeltasHist[0, 0, l] = DeltasHist[0, 0, l + 1];
                DeltasHist[0, 1, l] = DeltasHist[0, 1, l + 1];
            }

            DeltasHist[0, 0, in_ctx - 1] = delta.X;
            DeltasHist[0, 1, in_ctx - 1] = delta.Y;

            // Run inference
            using var ortInput = OrtValue.CreateTensorValueFromMemory(DeltasHist.ToArray(), in_shape_long);

            using var results = ort_session.Run(runOptions, new Dictionary<string, OrtValue> { { inputName, ortInput } }, new List<string> { outputName });
            var result = results.First()!.GetTensorDataAsSpan<float>();
            // Log.Debug("Neuropolator", "Result:");
            // for (int i = 0; i < 5; i++) Log.Debug("Neuropolator", string.Format("{0}, ", result[i]));
            // for (int i = 0; i < 5; i++) Log.Debug("Neuropolator", string.Format("{0}, ", result[i + 5]));

            foreach (int step in Enumerable.Range(0, stepsToTake))
            {
                report.Position += new Vector2(result[step], result[step + out_steps]) / scale;
            }

        }

        Emit?.Invoke(value);
    }

    // public void Dispose()
    // {
    //     ort_session.Dispose();
    // }
}
