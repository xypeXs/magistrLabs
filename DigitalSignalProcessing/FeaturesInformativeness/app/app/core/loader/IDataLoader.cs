using app.core.data;
using System;

namespace app.core.loader
{
    public interface IDataLoader
    {
        public bool isValidLoader(string fileName);
        public FeaturesData LoadData(string url);
    }
}
