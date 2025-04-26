using data;

namespace loader
{
    public interface IDataLoader
    {
        public FeaturesData LoadData(string url);
    }
}
