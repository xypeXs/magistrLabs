using informativeness.app.core.data;

namespace informativeness.app.core.loader
{
    public interface IDataLoader
    {
        public FeaturesData LoadData(string url);
    }
}
