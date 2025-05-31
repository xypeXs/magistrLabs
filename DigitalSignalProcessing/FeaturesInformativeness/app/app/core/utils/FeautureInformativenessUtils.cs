using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace app.core.utils
{
    public class FeautureInformativenessUtils
    {

        public const string FEATURE_NAME_REGEX_PATTERN = "[a-zA-Z]+";

        public static bool isStringContainsFeautureNames(string srcString)
        {
            if (srcString == null)
                return false;

            return Regex.Matches(srcString, FEATURE_NAME_REGEX_PATTERN).Count() > 0;
        }
    }
}
