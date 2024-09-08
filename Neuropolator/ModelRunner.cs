using Microsoft.ML.OnnxRuntime;
using NumSharp;
using System.Reflection;

namespace Neuropolator;

public class ModelRunner : IDisposable
{
    public ModelRunner(string modelName)
    {
        var modelPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, modelName);
        var sessionOptions = new SessionOptions
        {
            InterOpNumThreads = 1,
            IntraOpNumThreads = 1,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        _ortSession = new InferenceSession(modelPath, sessionOptions);
        _inputName = _ortSession.InputMetadata.First().Key;
        _outputName = _ortSession.OutputMetadata.First().Key;
        _inShape = _ortSession.InputMetadata[_inputName].Dimensions.ToArray();
        _inShapeLong = _inShape.Select(x => (long)x).ToArray();
        _outShape = _ortSession.OutputMetadata[_outputName].Dimensions.ToArray();
    }

    public NDArray Predict(NDArray inputDeltas)
    {
        using var ortInput = OrtValue.CreateTensorValueFromMemory(inputDeltas.ToArray<float>(), _inShapeLong);
        using var results = _ortSession.Run(_runOptions, new Dictionary<string, OrtValue> { { _inputName, ortInput } }, new List<string> { _outputName });
        var result = np.ndarray(_outShape, np.float32, results.First()!.GetTensorDataAsSpan<float>().ToArray());
        return result.reshape(new int[] { 2, _outSteps });
    }

    public NDArray PredictTransposed(NDArray inputDeltas) => Predict(inputDeltas.T).T;

    public int InCtx { get => _inCtx; }
    public int OutSteps { get => _outSteps; }

    private InferenceSession _ortSession;
    private string _inputName;
    private string _outputName;
    private int[] _inShape;
    private long[] _inShapeLong;
    private int _inCtx => _inShape[2];
    private int[] _outShape;
    private int _outSteps => _outShape[2];
    private RunOptions _runOptions = new();
    public void Dispose() => _ortSession.Dispose();
}
