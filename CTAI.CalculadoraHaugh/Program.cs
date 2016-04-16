using System;
using System.Collections.Generic; 
using System.Windows.Forms; 

namespace CTAI.CalculadoraHaugh
{
	class MainClass
	{
        [STAThread]
		public static void Main (string[] args)
		{
			Application.EnableVisualStyles(); 
			Application.SetCompatibleTextRenderingDefault(false); 
			Application.Run(new MainWindow());
		}
	}
}
