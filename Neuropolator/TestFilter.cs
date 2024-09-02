using System;
using System.Diagnostics;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using NumSharp;
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
        Log.Debug("Neuropolator", np.array(ort_session.InputMetadata[inputName].Dimensions).ToString());
    }
    public PipelinePosition Position => PipelinePosition.Raw;

    [Property("Space scale"), DefaultPropertyValue(0.02f)]
    public float spaceScale { get; set; }

    [Property("Steps"), DefaultPropertyValue(1), ToolTip("Number of steps to take, 0..5")]
    public int StepsToTake
    {
        get { return stepsToTake; }
        set { stepsToTake = System.Math.Clamp(value, 0, 5); }
    }
    private int stepsToTake;

    [Property("Step scale"), DefaultPropertyValue(1.0f)]
    public float stepScale { get; set; }

    [Property("Debug prediction spam"), DefaultPropertyValue(false)]
    public bool DebugPredictionSpam { get; set; }

    public event Action<IDeviceReport>? Emit;

    private static InferenceSession ort_session = new InferenceSession(
        Path.Combine(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location)!,
        "model.onnx"), new SessionOptions
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
    private static readonly int[] out_shape = ort_session.OutputMetadata[outputName].Dimensions;
    private static int out_steps = out_shape[2];
    private static readonly RunOptions runOptions = new RunOptions();

    private NDArray DeltasHist = np.zeros(in_shape, np.float32);
    private Vector2 previousPosition = Vector2.Zero;
    private HPETDeltaStopwatch reportStopwatch = new HPETDeltaStopwatch();


    public void Consume(IDeviceReport value)
    {
        if (value is IAbsolutePositionReport report)
        {
            // Reset logic
            if (reportStopwatch.Restart().TotalMilliseconds > 50)
            {
                DeltasHist = np.zeros(in_shape, np.float32);
                previousPosition = report.Position;
            }

            var delta = (report.Position - previousPosition) * spaceScale;
            previousPosition = report.Position;

            DeltasHist = DeltasHist.roll(-1, axis: 2);
            DeltasHist[0, 0, in_ctx - 1] = delta.X;
            DeltasHist[0, 1, in_ctx - 1] = delta.Y;

            // Run inference
            using var ortInput = OrtValue.CreateTensorValueFromMemory(DeltasHist.ToArray<float>(), in_shape_long);

            using var results = ort_session.Run(runOptions, new Dictionary<string, OrtValue> { { inputName, ortInput } }, new List<string> { outputName });

            var result = np.ndarray(out_shape, np.float32, results.First()!.GetTensorDataAsSpan<float>().ToArray());

            if (DebugPredictionSpam)
                Log.Debug("Neuropolator", "Result: " + result.ToString());

            foreach (int step in Enumerable.Range(0, stepsToTake))
            {
                float X = result.flat[step];
                float Y = result.flat[step + out_steps];
                report.Position += new Vector2(X, Y) * stepScale / spaceScale;
            }

        }

        Emit?.Invoke(value);
    }

    // public void Dispose()
    // {
    //     ort_session.Dispose();
    // }
}
