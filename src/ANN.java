import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class ANN {

	static double E = 2.71;
	static double N = 0.5;

	/*
	 * input.txt = inputs
	 * output.txt = round of hidden layers outs 
	 * readme.txt =	 hidden layers outs
	 */
	/*************************************
	 * Sigmoid Function
	 *************************************/

	public static double sigmoid(double net) {

		return (1 / (1 + Math.pow(E, -net)));
	}

	public static void main(String[] args) throws IOException {

		int Input[][] = new int[16][16];
		double Output_Net[][] = new double[16][16];
		double Output_Out[][] = new double[16][16];
		int Target_Output[][] = new int[16][16];

		double Input_Hidden_Between[][] = new double[16][4];
		double Hidden_Output_Between[][] = new double[4][16];

		double Hidden_Net[][] = new double[4][16];
		double Hidden_Out[][] = new double[4][16];
		double Temp_Error[] = new double[16];
		double Temp_Hidden_Output_Between[][] = new double[4][16];
		double Hiddens_Error[][] = new double[4][16];
		double Temp_Input_Hidden_Between[][] = new double[16][4];

		double error_total = 0.0;
		double temp = 0.0;
		int counter = 0;

		String line;
		String line_tmp[] = new String[2];

		FileReader File_Reader = new FileReader("input.txt");
		BufferedReader br = new BufferedReader(File_Reader);

		int i = 0;

		while ((line = br.readLine()) != null) {

			line_tmp = line.split(" ");

			for (int j = 0; j < 16; j++) {
				Input[i][j] = Integer.parseInt("" + line_tmp[0].charAt(j));
			}
			for (int j = 0; j < 16; j++) {
				Target_Output[i][j] = Integer.parseInt("" + line_tmp[0].charAt(j));
			}

			i++;
		}

		br.close();

		/*************************************
		 * Create Random Weights
		 *************************************/

		double upper = 0.5;
		double lower = -0.5;

		// random input_hidden_between values
		for (int l = 0; l < 16; l++) {
			for (int m = 0; m < 4; m++) {
				Input_Hidden_Between[l][m] = Math.random() * (upper - lower) + lower;
			}
		}
		// random hidden_output_between values
		for (int l = 0; l < 4; l++) {
			for (int m = 0; m < 16; m++) {
				Hidden_Output_Between[l][m] = Math.random() * (upper - lower) + lower;
			}
		}

		int dönücü = 0;
		while (dönücü < 10000) {

			/*************************************
			 * Find Hidden Layer Net Values
			 *************************************/

			while (counter < 16) {
				for (int a = 0; a < 4; a++) {
					for (int b = 0; b < 16; b++) {
						temp = Input[counter][b] * Input_Hidden_Between[b][a];
						Hidden_Net[a][counter] = temp + Hidden_Net[a][counter];
					}
				}
				counter++;
			}

			/*************************************
			 * Find Hidden Layer Out Sigmoid Values
			 *************************************/

			temp = 0.0;
			for (int a = 0; a < 4; a++) {
				for (int b = 0; b < 16; b++) {
					Hidden_Out[a][b] = sigmoid(Hidden_Net[a][b]);
				}
			}

			/*************************************
			 * Find Output Layer Net Values
			 *************************************/

			counter = 0;
			temp = 0.0;
			while (counter < 16) {
				for (int a = 0; a < 16; a++) {
					for (int b = 0; b < 4; b++) {
						temp = Hidden_Out[b][counter] * Hidden_Output_Between[b][a];
						Output_Net[counter][a] = temp + Output_Net[counter][a];
					}
				}
				counter++;
			}

			/*************************************
			 * Find Output Layer Out Sigmoid Values
			 *************************************/

			temp = 0.0;
			for (int a = 0; a < 16; a++) {
				for (int b = 0; b < 16; b++) {
					Output_Out[a][b] = sigmoid(Output_Net[a][b]);
				}
			}

			/*************************************
			 * Error Calculation - Output
			 *************************************/

			temp = 0.0;

			for (int a = 0; a < 16; a++) {
				error_total = 0;
				for (int b = 0; b < 16; b++) {
					temp = Math.pow((Target_Output[a][b] - Output_Out[a][b]), 2) / 2;
					error_total = error_total + temp;
				}
				Temp_Error[a] = error_total;
			}

			/*************************************
			 * Weight Calculation - Hidden_Output_Between
			 *************************************/
			counter = 0;
			while (counter < 16) {
				for (int a = 0; a < 16; a++) {
					for (int b = 0; b < 4; b++) {
						Temp_Hidden_Output_Between[b][a] = (-(Target_Output[counter][a] - Output_Out[counter][a]))
								* ((Output_Out[counter][a]) * (1 - Output_Out[counter][a])) * Hidden_Out[b][a];
					}
				}
				counter++;
			}

			/************************************
			 * Weight Update - Hidden_Output_Between
			 ************************************/

			counter = 0;
			while (counter < 4) {
				for (int k = 0; k < 16; k++) {
					Hidden_Output_Between[counter][k] = Hidden_Output_Between[counter][k]
							- (N * Temp_Hidden_Output_Between[counter][k]);
				}
				counter++;
			}

			/************************************
			 * Error Calculation - Hidden
			 ************************************/

			temp = 0.0;
			counter = 0;
			while (counter < 16) {
				for (int a = 0; a < 4; a++) {
					for (int b = 0; b < 16; b++) {
						temp = (-(Target_Output[counter][b] - Output_Out[counter][b]))
								* ((Output_Out[counter][b]) * (1 - Output_Out[counter][b]))
								* Hidden_Output_Between[a][b];
						Hiddens_Error[a][b] = temp + Hiddens_Error[a][b];
					}
				}
				counter++;
			}

			/************************************
			 * Weight Calculation - Input_Hidden_Between
			 ************************************/

			temp = 0.0;
			counter = 0;
			while (counter < 16) {
				for (int a = 0; a < 16; a++) {
					for (int b = 0; b < 4; b++) {
						Temp_Input_Hidden_Between[counter][b] = Hiddens_Error[b][a] * Hidden_Out[b][a]
								* (1 - Hidden_Out[b][a]) * Input[counter][a];
					}
				}
				counter++;
			}

			/************************************
			 * Weight Update - Input_Hidden_Between
			 ************************************/

			counter = 0;
			while (counter < 16) {
				for (int k = 0; k < 4; k++) {
					Input_Hidden_Between[counter][k] = Input_Hidden_Between[counter][k]
							- (N * Temp_Input_Hidden_Between[counter][k]);
				}
				counter++;
			}

			dönücü++;

		}

		String numString;

		for (int a = 0; a < 16; a++) {
			for (int b = 0; b < 4; b++) {
				System.out.print(String.format("%,.4f", Hidden_Out[b][a]) + " ");
			}
			System.out.println();
		}

		for (int a = 0; a < 16; a++) {
			for (int b = 0; b < 4; b++) {
				System.out.print(String.format(" " + Math.round(Hidden_Out[b][a])) + " ");
			}
			System.out.println();
		}

		File filee = new File("readme.txt");
		if (!filee.exists()) {
			filee.createNewFile();
		}

		FileWriter fileeWriter = new FileWriter(filee, false);
		BufferedWriter bWriterr = new BufferedWriter(fileeWriter);

		for (int a = 0; a < 16; a++) {
			for (int b = 0; b < 4; b++) {
				bWriterr.write(String.format("%,.4f", Hidden_Out[b][a]) + " ");
			}
			bWriterr.newLine();
		}

		bWriterr.close();

		File file = new File("output.txt");
		if (!file.exists()) {
			file.createNewFile();
		}

		FileWriter fileWriter = new FileWriter(file, false);
		BufferedWriter bWriter = new BufferedWriter(fileWriter);

		for (int a = 0; a < 16; a++) {
			for (int b = 0; b < 4; b++) {
				String Round = Double.toString(Math.round(Hidden_Out[b][a]));
				bWriter.write(Round + "\t");
			}
			bWriter.newLine();
		}

		bWriter.close();

	}
}