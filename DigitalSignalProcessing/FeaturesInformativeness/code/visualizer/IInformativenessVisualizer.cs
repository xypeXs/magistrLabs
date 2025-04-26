using code.data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace visualizer
{
    public interface IInformativenessVisualizer
    {
        public void visualize(InformativenessCalculationResult informativeness);
    }
}
