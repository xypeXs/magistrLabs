using core.data;

namespace core.loader
{
    public interface IDataLoader
    {
        public FeaturesData LoadData(string url);
    }
}
