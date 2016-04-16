using System;
using System.Drawing;
using System.Collections.Generic;

namespace CTAI.CalculadoraHaugh
{
	public class KmeansResult
	{
		public KmeansResult ()
		{
			properties = new Dictionary<string, object> ();
		}

		public Image FromImage {
			get;
			set;
		}

		public Image KmeansImage {
			get;
			set;
		}

		private Dictionary<String, object> properties;

		public object GetProperty(String Name){
			return properties [Name];
		}

		public void SetProperty(String Name, object Value){
			properties [Name] = Value;
		}
	}
}

