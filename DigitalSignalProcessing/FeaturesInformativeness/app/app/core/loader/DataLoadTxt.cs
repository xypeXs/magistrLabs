using app.core.data;
using System.IO;
using System.Linq;

namespace app.core.loader
{
    public class DataLoadTxt : IDataLoader
    {
        public bool isValidLoader(string fileName)
        {
            return fileName != null && fileName.EndsWith(".txt");
        }

        public FeaturesData LoadData(string url)
        {
            var data = new FeaturesData();
            var lines = File.ReadAllLines(url);
            foreach (var line in lines)
            {
                string[] parts = line.Split('\t').Select(x => x.Replace(".", ",")).Where(x => x != "").ToArray();
                var classLabel = int.Parse(parts[0]);
                var features = parts.Skip(1).Select(double.Parse).ToList();
                data.addImage(new Image { classIndex = classLabel, featureList = features });
            }
            return data;
        }
    }
}
