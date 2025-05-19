using app.core.data;
using DynamicData;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Text;

namespace app.core.loader
{
    public class DataLoadCsv : IDataLoader
    {
        protected string delimeter = ";";

        public bool isValidLoader(string fileName)
        {
            return fileName != null && fileName.EndsWith(".csv");
        }

        public FeaturesData LoadData(string url)
        {
            var data = new FeaturesData();

            string[] lines = File.ReadAllLines(url, Encoding.Default);

            int letterRegexCnt = Regex.Matches(lines[0], "[a-zA-Zа-яА-Я0-9\\s]*").Count();
            int dataStartInd = 0;
            if (letterRegexCnt != 0)
                dataStartInd = 1;

            int imageLength = 0;
            for (int i = dataStartInd; i < lines.Length; i++)
            {
                string[] parts = lines[i].Split(delimeter).Select(x => x.Replace(".", ",")).Where(x => x != "").ToArray();
                if (parts.Length == 0)
                    continue;

                imageLength = parts.Length;
                var classLabel = int.Parse(parts[0]);
                var features = parts.Skip(1).Select(double.Parse).ToList();
                data.addImage(new Image { classIndex = classLabel, featureList = features });
            }

            if (letterRegexCnt == 0)
                data.nameList = normalizeFeatureNames(new string[imageLength]);
            else
                data.nameList = normalizeFeatureNames(lines[0].Split(delimeter));

            return data;
        }

        protected List<string> normalizeFeatureNames(string[] featuresNamesFromFile)
        {
            List<string> featureNames = new List<string>(featuresNamesFromFile.Length);
            for (int i = 0; i < featuresNamesFromFile.Length; i++)
            {
                if (Regex.Matches(featuresNamesFromFile[i], "[a-zA-Zа-яА-Я0-9\\s]*").Count() == 0)
                    featureNames.Add("Feature #" + i);
                else
                    featureNames.Add(featuresNamesFromFile[i]);
            }

            return featureNames;
        }
    }
}