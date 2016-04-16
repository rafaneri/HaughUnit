using System; 
using System.Collections.Generic; 
using System.ComponentModel; 
using System.Data; 
using System.Drawing; 
using System.Text; 
using System.Windows.Forms; 
using System.Drawing.Imaging;
using NeuronDotNet.Core.Backpropagation; 
using System.IO;
using NeuronDotNet.Core;
using NeuronDotNet.Core.Initializers;
using System.Threading;
using NeuronDotNet.Core.LearningRateFunctions;
using System.Runtime.Serialization.Formatters.Binary;


namespace CTAI.CalculadoraHaugh
{
	public partial class MainWindow : Form 
	{
		private double eggWeight = 0.0d;
		private double learningRate = 0.4d;
		private int[] neuronCount = {4, 6, 4};
		private int cycles = 15000;
		private int cyclesBestParameters = 20;
		private int neuronsBestParameters = 20;
		private int layersBestParameters = 4;
		private BackpropagationNetwork network;
		private static int K = 4;// Número de K-médias
		private static int D = 5000;//A distância entre duas cores
		public Color[] z = new Color[K];//O vetor de k-médias

		public MainWindow ()
		{
			InitializeComponent();
//			LoadNetwork ();
			FindBestParameters ("/Volumes/MacintoshHDFiles/Documents/Mestrado/Inteligencia\\ Artificial/Material\\ Extra/Imagens\\ Ovos\\ Haugh/Trabalho/fotos\\ reais/treinamento");
		}

		private void open_Click(object sender, EventArgs e) 
		{ 
			OpenFileDialog openpicture = new OpenFileDialog(); 
			if (openpicture.ShowDialog() == DialogResult.OK) 
			{ 
				string[] cutPath = openpicture.FileName.Split ('_');
				eggWeight = double.Parse (cutPath[1]);
				pictureBox1.Image = Image.FromFile(openpicture.FileName); 
			} 
		} 

		private void exit_Click(object sender, EventArgs e) 
		{ 
			this.Close(); 
		} 

		//Calcula a distancia entre o RGB da cor c1 e a cor c2
		private int distance(Color c1, Color c2) 
		{ 
			int r1, r2, g1, g2, b1, b2, d; 
			r1 = c1.R; 
			r2 = c2.R; 
			g1 = c1.G; 
			g2 = c2.G; 
			b1 = c1.B; 
			b2 = c2.B; 
			d = (r1 - r2) * (r1 - r2) + (g1 - g2) * (g1 - g2) + (b1 - b2) * (b1 - b2); 
			return d; 
		} 

		private void train_Click(object sender, EventArgs e)
		{
			FolderBrowserDialog folderDialog = new FolderBrowserDialog(); 
			if (folderDialog.ShowDialog() == DialogResult.OK) 
			{ 
				string folder = folderDialog.SelectedPath;
				TrainNetwork (folder);
			} 
		}

		private void meanshift_Click(object sender, EventArgs e) 
		{
			if (eggWeight != 0.0d && network != null) {
				this.Cursor = Cursors.WaitCursor; 

				KmeansResult r = ClusterImage (pictureBox1.Image);

				this.Cursor = Cursors.Default; 
				pictureBox2.Refresh (); 
				pictureBox2.Image = r.KmeansImage;

				double height = (double)r.GetProperty ("Height");
				double saidaEsperada = 100 * Math.Log10(height - (1.7*Math.Pow(eggWeight, 0.37)) + 7.6);
				double saida = (network.Run (new double[] { height, eggWeight }) [0])*1000;

				label3.Text = String.Format ("UH esperada: {0:N2}, UH da rede: {1:N2}, altura da clara: {2:N2}mm, peso do ovo: {3}g", saidaEsperada, saida, r.GetProperty ("Height"), eggWeight);
			}else{
				MessageBox.Show("Nenhuma imagem foi selecionada.");
			}
		} 

		// Procura o melhor conjunto de treinamento para a rede
		private void FindBestParameters(string folder){
			ExecuteKmeansAndCreateTrainSet(folder, this.FindBestParametersNeuralNetwork);
		}

		// Realiza o treinamento da rede baseado no conjunto
		private void FindBestParametersNeuralNetwork(TrainingSet trainingSet){
			Dictionary<string, double> results = new Dictionary<string, double> ();
			string keyBestResult = string.Empty;
			double valueBestResult = double.MaxValue;
			BackpropagationNetwork bestNetwork = null;

			label3.Text = "Buscando a melhor configuração da rede";

			for (int i = 1; i<= cyclesBestParameters; i++) {
				int newCycle = 1000 * i;
				//int newNeurons = neuronCount * i;

				for (int j = 1; j <= neuronsBestParameters; j++) {
					if (!results.ContainsKey (newCycle + "|" + j)) {
						int[] layerInput = new int[layersBestParameters];
						for (int l = 0; l < layersBestParameters; l++) {
							if(l == 0 || l == layersBestParameters-1)
								layerInput [l] = j-1;
							else
								layerInput [l] = j;
						}

						ConfigureNetwork (layerInput);

						network.Learn (trainingSet, newCycle);
						network.StopLearning ();
						results.Add (newCycle + "|" + j, network.MeanSquaredError);
						Console.WriteLine (newCycle + "|" + j + " - " + network.MeanSquaredError);

						if (network.MeanSquaredError < valueBestResult) {
							valueBestResult = network.MeanSquaredError;
							keyBestResult = newCycle + "|" + j;
							bestNetwork = network;
						}
					}
				}
			}

			label3.Text = string.Format("O melhor conjunto de treinamento foi com (ciclos|neuronios) {0} gerando erro quadratico de {1}", keyBestResult, valueBestResult);
			Console.WriteLine (string.Format("O melhor conjunto de treinamento foi com (ciclos|neuronios) {0} gerando erro quadratico de {1}", keyBestResult, valueBestResult));

			// Configurar a rede com os novos parametros escolhidos baseado no melhor valor medido e salvar.
//			this.neuronCount = int.Parse(keyBestResult.Split ('|') [1]);
//			this.cycles = int.Parse(keyBestResult.Split ('|') [0]);
//			this.ConfigureNetwork (neuronCount);
//			this.TrainNeuralNetwork (trainingSet);
			network = bestNetwork;
			this.SaveNetwork();
		}

		// Realiza o treinamento da rede
		private void TrainNetwork(string folder){
			ConfigureNetwork (neuronCount);
			ExecuteKmeansAndCreateTrainSet(folder, this.TrainNeuralNetwork);
		}

		// Realiza configuração da rede
		private void ConfigureNetwork(int[] neurons){
			LinearLayer inputLayer = new LinearLayer(2);
			SigmoidLayer previousLayer = null;
			SigmoidLayer outputLayer = new SigmoidLayer(1);

			for (int i = 0; i < neurons.Length; i++) {
				SigmoidLayer hiddenLayer = new SigmoidLayer(neurons[i]);
				if (i == 0) {
					new BackpropagationConnector (inputLayer, hiddenLayer).Initializer = new RandomFunction (0d, learningRate);
				} else if (i == neurons.Length - 1) {
					new BackpropagationConnector (hiddenLayer, outputLayer).Initializer = new RandomFunction (0d, learningRate);
				} else {
					new BackpropagationConnector (previousLayer, hiddenLayer).Initializer = new RandomFunction (0d, learningRate);
				}

				previousLayer = hiddenLayer;
			}

			network = new BackpropagationNetwork(inputLayer, outputLayer);
			network.SetLearningRate(learningRate);
		}

		// Delegate para obter resultado do treinamento
		public delegate void ResponseTrainingSet(TrainingSet trainingSet);

		// Realiza a segmentação e gera o conjunto de treinamento
		private void ExecuteKmeansAndCreateTrainSet(string folder, ResponseTrainingSet response){
			TrainingSet trainingSet = new TrainingSet(2, 1);
			BackgroundWorker bw = new BackgroundWorker();
			string[] files = Directory.GetFiles(folder, "*.jpg", SearchOption.TopDirectoryOnly);

			bw.WorkerReportsProgress = true;
			bw.DoWork += new DoWorkEventHandler(
				delegate(object o, DoWorkEventArgs args)
				{
				BackgroundWorker b = o as BackgroundWorker;

				int index = 1;
				foreach (string fileName in files)
				{
					b.ReportProgress((index*100)/files.Length);
					KmeansResult result = ClusterImage (Image.FromFile (fileName));
					string[] cutPath = fileName.Split ('_');
					double weight = double.Parse (cutPath[1]);
					double height = (double)result.GetProperty ("Height");

					double saida = (100 * Math.Log10(height - (1.7*Math.Pow(weight, 0.37)) + 7.6))/1000;
					//Console.WriteLine (String.Format("Img: {0} peso: {1} altura: {2} UH: {3}", cutPath[0], weight, height, saida));

					trainingSet.Add(new TrainingSample(
						new double[] { height, weight }, 
					new double[] { saida }
					));

					index++;
				}
			});

			bw.ProgressChanged += new ProgressChangedEventHandler(
				delegate(object o, ProgressChangedEventArgs args)
				{
				label3.Text = string.Format("Segmentacao {0}% concluida", args.ProgressPercentage);
				trainingProgressBar.Value = args.ProgressPercentage;
			});

			bw.RunWorkerCompleted += new RunWorkerCompletedEventHandler(
				delegate(object o, RunWorkerCompletedEventArgs args)
				{
				label3.Text = "Segmentacao finalizada!";
				response(trainingSet);
			});

			bw.RunWorkerAsync ();
		}

		// Realiza o treinamento da rede baseado no conjunto
		private void TrainNeuralNetwork(TrainingSet trainingSet){
			network.EndEpochEvent += new TrainingEpochEventHandler(
				delegate(object senderNetwork, TrainingEpochEventArgs args)
				{
				label3.Text = string.Format("Treinamento {0}% concluido", trainingProgressBar.Value);
				trainingProgressBar.Value = (int)(args.TrainingIteration * 100d / cycles);
				Application.DoEvents();
			});

			label3.Text = "Iniciando treinamento da rede...";
			network.Learn(trainingSet, cycles);
			network.StopLearning();
			this.SaveNetwork();
			label3.Text = "Treinamento finalizado. Erro quadratico: " + network.MeanSquaredError;
		}

		// Carregar a rede treinada
		private void LoadNetwork(){
			string filename = Directory.GetCurrentDirectory() + "/net.nn";
			if (File.Exists (filename)) {
				Stream stream = File.Open (filename, FileMode.Open); 
				BinaryFormatter formatter = new BinaryFormatter (); 
				network = (BackpropagationNetwork)formatter.Deserialize (stream); 
				stream.Close ();

				label3.Text = "Existe uma rede treinada e carregada.";
			} else {
				label3.Text = "Nao existe nenhuma rede treinada e carregada.";
			}
		}

		// Salvar a rede treinada
		private void SaveNetwork(){
			Console.WriteLine ("Erro quadratico: " + network.MeanSquaredError);
			string filename = Directory.GetCurrentDirectory() + "/net.nn";
			if (File.Exists (filename))
				File.Delete (filename);
			Stream stream = File.Open(filename, FileMode.Create); 
			BinaryFormatter formatter = new BinaryFormatter(); 
			formatter.Serialize(stream, network); 
			stream.Close();
		}

		private KmeansResult ClusterImage(System.Drawing.Image SampleImage)
		{
			Bitmap before = new Bitmap (SampleImage);
			Bitmap after = new Bitmap(SampleImage.Width, SampleImage.Height);

			Color c1 = new Color(); 
			Color follow = System.Drawing.Color.FromArgb (255, 140, 100, 70);
			int i,j,k,tot,old,label,dis,temp,N,compare,rr,gg,bb,W,H,R,G,B; 
			W = SampleImage.Width; 
			H = SampleImage.Height; 
			//Console.WriteLine(W); 
			//Console.WriteLine(H); 
			int[][] type = new int[W][];// Cria uma matriz W x H onde em cada dimensão W existirá um vetor de H posições
			for (i = 0; i < W; i++)  // Varre de 0 até W
				type[i] = new int[H]; // Cria para a posição i do vetor W um vetor de H posições

			//for (i = 0; i < W - 1; i++)
			//	for (j = 0; j < H - 1; j++) 
			//		type[i][j] = 0; 
			c1 = before.GetPixel(0,0); 
			z[0] = c1; 
			tot = 1; 
			temp = 1; 
			for (i = 0; i < W - 1; i++)            //Percorre os pixels da imagem em X
			{ 
				for (j = 1; j < H - 1; j++) 
				{ 
					c1 = before.GetPixel(i, j); 
					label = 1; 
					for (k = 0; k < tot; k++) 
					{ 
						dis = distance(c1, z[k]);
						if (dis < D)  label = 0;    //Se a distancia entre o RGB das cores for menor que D 
						//o label será 0, logo a cor permanece no mesmo grupo
					} 
					if (label == 1) // Caso seja uma nova cor
					{ 
						if (tot == K) // verificar se o total de passadas é igual a K 
							// e atualiza uma variavel de controle para sair do laço
						{ 
							temp = -1; 
							break; 
						} 
						z[tot] = c1; // adiciona a nova cor para o vetor de cores z
						tot++; 
					} 
				} 
				if (temp == -1) break; 
			} 
			//Console.WriteLine(tot); 
			do                                     //è¿­ä»£ 
			{ 
				compare = 0; 
				for (i = 0; i < W - 1; i++) 
				{ 
					for (j = 0; j < H - 1; j++) 
					{ 
						c1 = before.GetPixel(i, j); 
						temp = 0; 
						for (k = 0; k < tot; k++) 
						{ 
							if (distance(c1, z[temp]) > distance(c1, z[k])) temp = k; 
						} 
						old = type[i][j];          //åˆ¤æ–­åˆ†ç±»æœ‰æ²¡æœ‰å˜åŒ– 
						type[i][j] = temp; 
						if (old != temp) compare = 1; 
					} 
				} 

				for (k = 0; k < tot; k++) 
				{             
					//è®¡ç®—æ–°èšç±»ä¸­å¿ƒ 
					R = 0; 
					G = 0; 
					B = 0; 
					N = 0; 
					for (i = 0; i < W - 1; i++) 
					{ 
						for (j = 0; j < H - 1; j++) 
						{ 
							c1 = before.GetPixel(i, j); 
							if (type[i][j] == k) 
							{ 
								R += c1.R; 
								G += c1.G; 
								B += c1.B; 
								N += 1; 
							} 
						} 
					} if (N != 0) 
					{ 
						R = R / N; 
						G = G / N; 
						B = B / N; 
						z[k] = Color.FromArgb(R, G, B); 
					} 
				} 
				if (compare == 0) break; 
			} 
			while (true);

			Color older = Color.Transparent;
			int sIndex = 0;
			int index = 0;
			foreach (var item in z) {
				dis = distance(follow, item);
				//Console.WriteLine (dis);
				if (dis < D && (older == Color.Transparent || distance(item, older) < dis)){
					older = item;
					sIndex = index;
				}
				index++;
			}
			//Console.WriteLine (older.ToString());
			//Console.WriteLine (z[sIndex].ToString());

			// Reconstrói a imagem com a cor centróide de cada k-média
			int posH = H - 1;
			int posW = 0;
			for (i = 0; i < W - 1; i++) 
			{ 
				for (j = 0; j < H - 1; j++) 
				{
					rr = z[type[i][j]].R;
					gg = z[type[i][j]].G;
					bb = z[type[i][j]].B;
					after.SetPixel(i, j, Color.FromArgb(rr, gg, bb));

					// Obtendo a maior altura em PX
					if(sIndex == type[i][j]){
						if(j < posH) {
							if(H-j > 4){

								Color actual = after.GetPixel (i, j);
								bool isEquals = true;

								// Checa se pelo menos 4px abaixo são da mesma cor
								for(int nJ = j+1; nJ < j+4; nJ++){

									rr = z[type[i][nJ]].R;
									gg = z[type[i][nJ]].G;
									bb = z[type[i][nJ]].B;

									if (actual.R != rr || actual.G != gg || actual.B != bb){
										isEquals = false;
										break;
									}

								}

								// Checa se pelo menos 4px da direita são da mesma cor
								for(int nWD = i+1; nWD < i+4; nWD++){

									if(nWD < W){
										rr = z[type[nWD][j]].R;
										gg = z[type[nWD][j]].G;
										bb = z[type[nWD][j]].B;

										if (actual.R != rr || actual.G != gg || actual.B != bb){
											isEquals = false;
											break;
										}
									}
								}

								// Checa se pelo menos 4px da esquerda são da mesma cor
								for(int nWD = i-1; nWD > i-4; nWD--){

									if(nWD > 0){
										rr = z[type[nWD][j]].R;
										gg = z[type[nWD][j]].G;
										bb = z[type[nWD][j]].B;

										if (actual.R != rr || actual.G != gg || actual.B != bb){
											isEquals = false;
											break;
										}
									}

								}

								if(isEquals){
									posH = j;
									posW = i;
								}
							}
						}
						//Console.WriteLine (type[i][j] + " i: " + i + " j: " + j );
						//Console.WriteLine ("Altura em px = " + (H-j));
					}
				} 
			} 

			//Console.WriteLine (String.Format ("Altura: {0}mm, [x,y]: [{1}, {2}], cor: {3}", (((H-posH)*2.71)/11), posW, posH, z[sIndex].ToString()));
			//
			after.SetPixel(posW, posH, Color.Red);
			KmeansResult result = new KmeansResult ();
			result.FromImage = SampleImage;
			result.KmeansImage = after;
			result.SetProperty ("Height", (((H-posH)*2.71)/11));
			result.SetProperty ("X", posW);
			result.SetProperty ("Y", posH);
			result.SetProperty ("Color", z[sIndex]);
			return result;
		}

		private System.ComponentModel.IContainer components = null; 

		protected override void Dispose(bool disposing) 
		{ 
			if (disposing && (components != null)) 
			{ 
				components.Dispose(); 
			} 
			base.Dispose(disposing); 
		} 

		#region Windows  

		/// <summary> 
		/// è®¾è®¡å™¨æ”¯æŒæ‰€éœ€çš„æ–¹æ³• - ä¸è¦ 
		/// ä½¿ç”¨ä»£ç ç¼–è¾‘å™¨ä¿®æ”¹æ­¤æ–¹æ³•çš„å†…å®¹ã€‚ 
		/// </summary> 
		private void InitializeComponent() 
		{ 
			this.pictureBox1 = new System.Windows.Forms.PictureBox(); 
			this.pictureBox2 = new System.Windows.Forms.PictureBox();
			this.label1 = new Label ();
			this.label2 = new Label ();
			this.label3 = new Label ();
			this.open = new System.Windows.Forms.Button(); 
			this.exit = new System.Windows.Forms.Button(); 
			this.meanshift = new System.Windows.Forms.Button(); 
			this.train = new System.Windows.Forms.Button(); 
			this.trainingProgressBar = new System.Windows.Forms.ProgressBar();
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit(); 
			((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit(); 
			this.SuspendLayout(); 

			//
			// label1
			//
			this.label1.Location = new System.Drawing.Point(20, 80); 
			this.label1.Size = new System.Drawing.Size(600, 22); 
			this.label1.Text = "Imagem Selecionada:";

			//
			// label2
			//
			this.label2.Location = new System.Drawing.Point(20, 220); 
			this.label2.Size = new System.Drawing.Size(600, 22); 
			this.label2.Text = "Imagem Segmentada:";

			//
			// label3
			//
			this.label3.Location = new System.Drawing.Point(20, 360); 
			this.label3.Size = new System.Drawing.Size(600, 22); 

			//  
			// pictureBox1 
			//  
			this.pictureBox1.Location = new System.Drawing.Point(20, 100); 
			this.pictureBox1.Name = "pictureBox1"; 
			this.pictureBox1.Size = new System.Drawing.Size(590, 105); 
			//this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage; 
			this.pictureBox1.TabIndex = 0; 
			this.pictureBox1.TabStop = false; 
			//  
			// pictureBox2 
			//  
			this.pictureBox2.Location = new System.Drawing.Point(20, 240); 
			this.pictureBox2.Name = "pictureBox2"; 
			this.pictureBox2.Size = new System.Drawing.Size(590, 105); 
			//this.pictureBox2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage; 
			this.pictureBox2.TabIndex = 1; 
			this.pictureBox2.TabStop = false; 
			//  
			// open 
			//  
			this.open.Location = new System.Drawing.Point(20, 20); 
			this.open.Name = "open"; 
			this.open.Size = new System.Drawing.Size(160, 28); 
			this.open.TabIndex = 2; 
			this.open.Text = "Selecionar Imagem";
//			this.open.Text = "Selecionar Imagem"; 
			this.open.UseVisualStyleBackColor = true; 
			this.open.Click += new System.EventHandler(this.open_Click); 
			//  
			// exit 
			//  
			this.exit.Location = new System.Drawing.Point(560, 20); 
			this.exit.Name = "exit"; 
			this.exit.Size = new System.Drawing.Size(40, 28); 
			this.exit.TabIndex = 3; 
			this.exit.Text = "Sair"; 
			this.exit.UseVisualStyleBackColor = true; 
			this.exit.Click += new System.EventHandler(this.exit_Click); 
			//  
			// meanshift 
			//  
			this.meanshift.Location = new System.Drawing.Point(200, 20); 
			this.meanshift.Name = "meanshift"; 
			this.meanshift.Size = new System.Drawing.Size(160, 28); 
			this.meanshift.TabIndex = 4; 
			this.meanshift.Text = "Medir imagem"; 
			this.meanshift.UseVisualStyleBackColor = true; 
			this.meanshift.Click += new System.EventHandler(this.meanshift_Click); 
			//
			// train
			//
			this.train.Location = new System.Drawing.Point(380, 20); 
			this.train.Name = "trains"; 
			this.train.Size = new System.Drawing.Size(160, 28); 
			this.train.TabIndex = 5; 
			this.train.Text = "Executar treinamento"; 
			this.train.UseVisualStyleBackColor = true; 
			this.train.Click += new System.EventHandler(this.train_Click); 
			// 
			// trainingProgressBar
			// 
			this.trainingProgressBar.Location = new System.Drawing.Point(20, 385);
			this.trainingProgressBar.Name = "trainingProgressBar";
			this.trainingProgressBar.Size = new System.Drawing.Size(600, 20);
			this.trainingProgressBar.TabIndex = 6;
			//  
			// Form1 
			//  
			//this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F); 
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font; 
			this.ClientSize = new System.Drawing.Size(640, 425); 
			this.Controls.Add(this.train); 
			this.Controls.Add(this.meanshift); 
			this.Controls.Add(this.exit); 
			this.Controls.Add(this.open); 
			this.Controls.Add(this.pictureBox2); 
			this.Controls.Add(this.pictureBox1); 
			this.Controls.Add(this.label1); 
			this.Controls.Add(this.label2); 
			this.Controls.Add(this.label3); 
			this.Controls.Add(this.trainingProgressBar); 
			this.Name = "Form1"; 
			this.Text = "Calculadora Unidade Haugh"; 
			((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit(); 
			((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit(); 
			this.ResumeLayout(false); 

		} 

		#endregion 

		private System.Windows.Forms.PictureBox pictureBox1; 
		private System.Windows.Forms.PictureBox pictureBox2; 
		private System.Windows.Forms.Button open; 
		private System.Windows.Forms.Button exit; 
		private System.Windows.Forms.Button meanshift;
		private System.Windows.Forms.Button train;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.ProgressBar trainingProgressBar;

	}
}

