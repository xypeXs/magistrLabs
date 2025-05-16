using informativeness.app.core.data;

namespace informativeness.app.core.visualizer
{
    public interface IInformativenessVisualizer
    {
        public void visualize(InformativenessCalculationResult informativeness, string outputPath, string name);
    }
}