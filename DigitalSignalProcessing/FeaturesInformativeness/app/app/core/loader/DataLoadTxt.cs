using app.core.data;
using app.core.utils;
using System.Collections.Generic;
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
            data.nameList = GenerateFeatureNames(data.imageList[0].featureList.Count + 1);
            return data;
        }

        protected List<string> GenerateFeatureNames(int count)
        {
            List<string> featureNames = new List<string>(count);
            for (int i = 0; i < count; i++)
            {
                featureNames.Add("Feature #" + i);
            }

            return featureNames;
        }
    }
}
