using NumSharp;
using OpenTabletDriver.Plugin;
using OpenTabletDriver.Plugin.Attributes;
using OpenTabletDriver.Plugin.Output;
using OpenTabletDriver.Plugin.Tablet;
using OpenTabletDriver.Plugin.Timing;
using System.Numerics;

namespace Neuropolator;

public class TestFilter : IPositionedPipelineElement<IDeviceReport>
{
    public TestFilter() : base() { }
    public PipelinePosition Position => PipelinePosition.Raw;

    [Property("Space scale"), DefaultPropertyValue(0.02f)]
    public float SpaceScale { get; set; }

    [Property("Steps"), DefaultPropertyValue(1), ToolTip("Number of steps to take, 0..5")]
    public int StepsToTake
    {
        get { return _stepsToTake; }
        set { _stepsToTake = Math.Clamp(value, 0, 5); }
    }
    private int _stepsToTake;

    [Property("Step scale"), DefaultPropertyValue(1.0f)]
    public float StepScale { get; set; }

    [Property("Debug prediction spam"), DefaultPropertyValue(false)]
    public bool DebugPredictionSpam { get; set; }

    [Property("Reset time"), Unit("ms"), DefaultPropertyValue(50.0f)]
    public float ResetTime { get; set; }

    public event Action<IDeviceReport>? Emit;

    private static ModelRunner _runner = new("model.onnx");
    private static int _inCtx = _runner.InCtx;
    private static int[] _inShape = new[] { 2, _inCtx };

    private NDArray _deltasHist = np.zeros(_inShape, np.float32);
    private Vector2 _previousPosition = Vector2.Zero;
    private HPETDeltaStopwatch _reportStopwatch = new HPETDeltaStopwatch();
    private HPETDeltaStopwatch _strokeStopwatch = new HPETDeltaStopwatch();


    public void Consume(IDeviceReport value)
    {
        if (value is IAbsolutePositionReport report)
        {
            // Reset
            if (_reportStopwatch.Restart().TotalMilliseconds > ResetTime)
            {
                _strokeStopwatch.Restart();
                _previousPosition = report.Position;
                _deltasHist = np.zeros(_inShape, np.float32);
            }

            var delta = (report.Position - _previousPosition) * SpaceScale;
            _previousPosition = report.Position;

            _deltasHist = _deltasHist.roll(-1, axis: 1);
            _deltasHist[0, _inCtx - 1] = delta.X;
            _deltasHist[1, _inCtx - 1] = delta.Y;

            var result = _runner.Predict(_deltasHist);

            if (DebugPredictionSpam)
                Log.Debug("Neuropolator", "Result: " + result.ToString());

            foreach (int step in Enumerable.Range(0, _stepsToTake))
            {
                float X = result[0, step];
                float Y = result[1, step];
                report.Position += new Vector2(X, Y) * StepScale / SpaceScale;
            }
        }

        Emit?.Invoke(value);
    }
}
