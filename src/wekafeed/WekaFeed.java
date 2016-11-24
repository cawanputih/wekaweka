
package wekafeed;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Scanner;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instance;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.Filter;

public class WekaFeed extends AbstractClassifier {
  
  /* After reading model */
  public Node[][] neuralNode_read;
  
  /* Node attributes */
  public Node[][] neuralNode;
  public static long seed = System.currentTimeMillis();
  public static Random rand = new Random(seed);
  public static double learningrate=1;
  public static int nkelas; // atribut ini ada untuk tes saja
  public static int indexClass; // atribut ini ada untuk indeks kelas
  
  
  public int inputCount;
  public int hiddenLayerCount;
  public int hiddenCount;
  public int outputCount;
  public int totalLayer;
  
  public static int numOfIteration;
  
  
  /* Used model attributes */
  public static int useModel = 0;

  
//==============================================================================
  
  // Contructor without arguments
  WekaFeed() {
      
  }
  
//==============================================================================
  
  // Constructor with arguments
  WekaFeed(int inputCount, int hiddenLayerCount, int hiddenCount, 
          int outputCount){
      
    // initialize the attributes
    this.inputCount = inputCount;
    this.hiddenLayerCount = hiddenLayerCount;
    this.hiddenCount = hiddenCount;
    this.outputCount = outputCount;
    
    if (hiddenLayerCount == 0) {
        totalLayer = 2;
    } else {
        totalLayer = 3;
    }
            
    
    int layerCount = 0;  
    neuralNode = new Node[2+hiddenLayerCount][];
    
    //input Node
    neuralNode[layerCount] = new Node[inputCount];
    for(int j=0; j<inputCount; j++){
      neuralNode[layerCount][j] = new Node();
    }
    layerCount++;
    
    //hidden Node
    for(int i=0; i<hiddenLayerCount; i++){
      neuralNode[layerCount] = new Node[hiddenCount];
      for(int j=0; j<hiddenCount; j++){
        neuralNode[layerCount][j] = new Node();
      }
      layerCount++;
    }
    
    //output Node
    neuralNode[layerCount] = new Node[outputCount];
    for(int j=0; j<outputCount; j++){
      neuralNode[layerCount][j] = new Node();
    }
    layerCount++;
    
    connectNodes();
    
    nkelas=outputCount;
    
  }
//==============================================================================
  public void connectNodes(){
    int layerCount = neuralNode.length;
    int layerNodes;
    int nextLayerNodes;
    for(int i=0; i<layerCount-1; i++)
    {
      layerNodes = neuralNode[i].length;
      nextLayerNodes = neuralNode[i+1].length;
      for(int j=0; j<layerNodes; j++)
      {
        for(int k=0; k<nextLayerNodes; k++)
        {
             
            // use random values as the weights
            neuralNode[i][j].edges.put(neuralNode[i+1][k].id, rand.nextDouble());
        
        }
      }
    }
  }
//==============================================================================
  public void printAllEdge(){
    int layerCount = neuralNode.length;
    int layerNodes;
    for(int i=0; i<layerCount; i++)
    {
      System.out.println("Layer"+i+"============================");
      layerNodes = neuralNode[i].length;
      for(int j=0; j<layerNodes; j++)
      {
        System.out.println("Node"+neuralNode[i][j].id+"----------------------");
        for(int key: neuralNode[i][j].edges.keySet()){
          System.out.println("ID: "+key+" value: "+neuralNode[i][j].edges.get(key));
        }
      }
    }
  }
//==============================================================================
  public void printAllNode(){
    int layerCount = neuralNode.length;
    int layerNodes;
    for(int i=0; i<layerCount; i++)
    {
      System.out.println("Layer"+i+"========================");// + getLayerSigma(i));
      layerNodes = neuralNode[i].length;
      for(int j=0; j<layerNodes; j++)
      {
        System.out.println(neuralNode[i][j].id+": "+neuralNode[i][j].value);
        System.out.println("Eror: "+neuralNode[i][j].eror);
      }
    }
  }
//==============================================================================
  public double getLayerSigma(int layerIndex){
    int layerNodes = neuralNode[layerIndex].length;
    double sum = 0;
    for(int j=0; j<layerNodes; j++)
    {
      sum += neuralNode[layerIndex][j].value;
    }
    
    return sum;
  }
//==============================================================================
  public boolean assignInput(double[] initiator){
    if(neuralNode[0] != null)
    {
      if(neuralNode[0].length == initiator.length)
      {
        for(int i=0;i<neuralNode[0].length; i++)
        {
          neuralNode[0][i].value = initiator[i];
        }
      }
      else
      {
        System.out.println("Banyak nilai salah");
        return false;
      }
    }
    else
    {
      System.out.println("eror");
      return false;
    }
    
    return true;
  }
//==============================================================================
  public int[] searchNode(int searchId){
    int[] result = new int[2];
    result[0] = -1;
    result[1] = -1;
    
    for(int i=0; i<neuralNode.length; i++){
      for(int j=0; j<neuralNode[i].length; j++){
        if(neuralNode[i][j].id == searchId){
          result[0] = i;
          result[1] = j;
        }
      }
    }
    return result;
  }
//==============================================================================
  
  // mengembalikan id yang dimiliki oleh suatu neuralNode
  public int getidnode(Node x)
  {   
      return x.id;
  }
  
//==============================================================================  
  public boolean assignPostEdgeWeight(int id, double[] initiator){
    int[] index = searchNode(id);
    int i = index[0];
    int j = index[1];
    int k = 0;
    
    if(initiator.length != neuralNode[i].length)
    {
      System.out.println("kurang panjang");
      return false;
    }
    
    for(int key: neuralNode[i][j].edges.keySet()){
      neuralNode[i][j].edges.put(key, initiator[k]);
      k++;
    }
    
    return true;
  }
//==============================================================================  
  public boolean assignPreEdgeWeight(int id, double[] initiator){
    int[] index = searchNode(id);
    int i = index[0];
    int j = index[1];
    int k = 0;
    int prevLayer = i-1;
    
    if(i<1)
    {
      System.out.println("oh tidak bisa");
      return false;
    }    

    if(initiator.length != neuralNode[prevLayer].length)
    {
      System.out.println("kurang panjang");
      return false;
    }
    
    for(j=0; j<neuralNode[prevLayer].length; j++){
      for(int key: neuralNode[prevLayer][j].edges.keySet()){
        if(key == id){
          neuralNode[prevLayer][j].edges.put(key, initiator[k]);
          k++;
        }
      }
    }
    
    return true;
  }
//==============================================================================  
  public boolean assignEdge(int idpre, int idpost, double initiator){
    int[] indexpre = searchNode(idpre);
    int ipre = indexpre[0];
    int jpre = indexpre[1];
    
    for(int key: neuralNode[ipre][jpre].edges.keySet()){
      if(key == idpost){
        neuralNode[ipre][jpre].edges.put(key, initiator);
        return true;
      }
    }
    
    
    return false;
  }
//==============================================================================
  public double sigmaNode(int id){
    int[] index = searchNode(id);
    int i = index[0];
    int j = index[1];
    double sum = 0;
    double weight = 0;
    
    int ii = i-1;
    for(int jj=0; jj< neuralNode[ii].length; jj++){
      weight = 0;
      for(int key: neuralNode[ii][jj].edges.keySet()){
        if(key == id){
          weight = neuralNode[ii][jj].edges.get(key);
          sum += neuralNode[ii][jj].value * weight;
        }
      }
      
      
    }
    
    return sum;
  }
//==============================================================================
  // Set value baru untuk neuralNode yang mempunyai suatu id tertentu
    public void setvaluebaru(int id){
		double negatifnet = -1*sigmaNode(id);
		double hasilexp;
		int[] indeks = searchNode(id);
		
		
		hasilexp=1/(1+Math.exp(negatifnet));
        neuralNode[indeks[0]][indeks[1]].value=hasilexp;        
        //System.out.println(indeks[0]+" "+indeks[1]);
    }

//==============================================================================
	// melakukan penghitungan eror untuk setiap node pada layer output
	public void hitungeroroutput(int id, double target){
		double keluaran;
		double hitungeror;
		int[] indeks = searchNode(id);
		
		keluaran = neuralNode[indeks[0]][indeks[1]].value;
		
		hitungeror=keluaran*(1-keluaran)*(target-keluaran);
		neuralNode[indeks[0]][indeks[1]].eror=hitungeror;  
    }

//==============================================================================
	// melakukan penghitungan sigma seluruh eror dari node yang dituju oleh node id
	// dikalikan dengan weight dari node id menuju node tersebut
	public double sigmaerorkaliweight(int id){
		
		int[] index = searchNode(id);
		int[] index2;
		int i = index[0];
		int j = index[1];
		double sum = 0;
		double weight = 0;
		
		  for(int key: neuralNode[i][j].edges.keySet()){
			  weight = neuralNode[i][j].edges.get(key);
			  index2 =searchNode(key);
			  sum += neuralNode[index2[0]][index2[1]].eror * weight;
		  }
		  
		return sum;
	}

//==============================================================================
	// melakukan penghitungan eror untuk setiap node pada setiap layer hidden yang ada
	public void hitungerorhidden(int id){
		double keluaran;
		double hitungeror;
		int[] indeks = searchNode(id);
		
		keluaran = neuralNode[indeks[0]][indeks[1]].value;
		
		hitungeror=keluaran*(1-keluaran)*(sigmaerorkaliweight(id));
		neuralNode[indeks[0]][indeks[1]].eror=hitungeror;  
    }

//==============================================================================
	
	// melakukan pengubahan bobot yang berasal dari idnodeasal menuju ke idnodetujuan
	// learning rate di set di constructor dengan nilai satu
	public void ubahbobot(int idnodeasal, int idnodetujuan){
		int[] indeksasal = searchNode(idnodeasal);
		int[] indekstujuan = searchNode(idnodetujuan);
		
		double bobotawal = neuralNode[indeksasal[0]][indeksasal[1]].edges.get(idnodetujuan);
		double bobotbaru = bobotawal + neuralNode[indekstujuan[0]][indekstujuan[1]].eror*neuralNode[indeksasal[0]][indeksasal[1]].value*learningrate;
		
                
                
		neuralNode[indeksasal[0]][indeksasal[1]].edges.put(idnodetujuan,bobotbaru);
	}

//==============================================================================
    public double[] neuralNodeOutput(){  
    Node[] outputNode = neuralNode[neuralNode.length-1];
    double[] output = new double[outputNode.length];
    
    for(int i=0; i<outputNode.length; i++){
      output[i] = outputNode[i].value;
    }
    
    return output;
  }
//==============================================================================
  public int getFirstnonInput(){
    return neuralNode[0].length;
  }
//==============================================================================

// Melakukan feed forward yang dimulai dari "idmulaiassign" =>  node input tidak dapat dicari value barunya
  public void feedforward(int idmulaiassign){
	int banyaknode = Node.lastID;
        
        for(int i=idmulaiassign; i< banyaknode ; i++)
        {
            setvaluebaru(i);
        }
  }
//==============================================================================

// Melakukan feed forward yang dimulai dari "idmulaiassign" =>  node input tidak dapat dicari value barunya
  public void feedforward(){
  int idmulaiassign = getFirstnonInput();
	int banyaknode = Node.lastID;
        
        for(int i=idmulaiassign; i< banyaknode ; i++)
        {
            setvaluebaru(i);
        }
  }
//==============================================================================

// Melakukan back propagation. Double[] target diassign dengan target seharusnya dikeluaran
// target[0] sebagai target pada node 0 output layer
// target[1] sebagai target pada node 1 output layer
// target[2] sebagai target pada node 2 output layer
// ...
  public void backpropagation(double[] target){  
	  
	    
	// Set eror untuk setiap node yang ada pada layer output
	    
	int xx=nkelas-1;
	int banyaklayer=neuralNode.length;
        int qq = banyaklayer-1;
        int yy=neuralNode[qq].length;
        int aa = yy-1;
		
        for (int jj = aa ; jj>=0 ; jj--){
                hitungeroroutput(neuralNode[qq][jj].id,target[xx]);
                xx--;
        }
        
        
	// Set eror dan juga bobot baru untuk setiap node yang ada pada masing-masing layer hidden
	xx = banyaklayer-2;
	
	for(int ii = xx ; ii> 0 ; ii--){
		yy=neuralNode[ii].length;
		int zz=neuralNode[ii+1].length;
		
		aa = yy-1;
		int bb = zz-1;
		
		for (int jj = aa ; jj>=0 ; jj--){
			for ( int kk = bb ; kk >=0 ; kk--) {
				ubahbobot(neuralNode[ii][jj].id,neuralNode[ii+1][kk].id);
			}
		}
		
		for (int jj = aa ; jj>=0 ; jj--){
			hitungerorhidden(neuralNode[ii][jj].id);
		}
		
	}
	
	// Set bobot baru yang berasal dari layer input
		yy=neuralNode[0].length;
		int zz=neuralNode[1].length;
		
		aa = yy-1;
		int bb = zz-1;
		
		for (int jj = aa ; jj>=0 ; jj--){
				for ( int kk = bb ; kk >=0 ; kk--) {
					ubahbobot(neuralNode[0][jj].id,neuralNode[1][kk].id);
				}
		}
		
  }  

//==============================================================================
  @Override
  public void buildClassifier(Instances data) throws Exception {
    int index = 0;
    int banyakAtribut = data.numAttributes()-1;
    int banyakKelas = data.numClasses();
    int banyakData = data.numInstances();
    
    
    for (int idx0 = 0; idx0 < numOfIteration; idx0++) {
    
    for(index=0; index<banyakData; index++)
    {
      //pilih instance
      Instance curr = data.get(index);
      
      
      //create initial value
      //System.out.println("input===============");
      double[] input = new double[banyakAtribut];
      int j=0;
      for(int i=0; i<banyakAtribut+1; i++){
        if(i!=indexClass){
        input[j] = curr.value(i);
		j++;
		}
        //System.out.println(input[i]);
      }

      //create target
      //System.out.println("target==============");
     // System.out.println(curr.classValue());
      double[] target = new double[21]; //anggap inisialisasi 0
      int indexKelas = (int) curr.classValue();
      target[indexKelas] = 1;
      for(int i=0; i<banyakKelas; i++){
        //System.out.println("kelas"+i+": "+target[i]);
      }
      

      //jalankan
      assignInput(input);
      feedforward();
      backpropagation(target);
    }
    
    }
  }  
//==============================================================================
  
  @Override
  public double[] distributionForInstance(Instance instance)
                                 throws java.lang.Exception{
     

    int banyakAtribut = instance.numAttributes()-1;
    double[] input = new double[banyakAtribut];
    
    int j=0;
      for(int i=0; i<banyakAtribut+1; i++){
        if(i!=indexClass){
        input[j] = instance.value(i);
		j++;
		}
        //System.out.println(input[i]);
      }
    /*  
    for(int i=0; i<banyakAtribut; i++){
      input[i] = instance.value(i);
      //System.out.println("input: "+input[i]);
    }
    */
    assignInput(input);
    
    feedforward();
    
    double[] output = neuralNodeOutput();
    for(int i=0; i<3; i++){
//        System.out.println("kelas"+i+": "+input[i]);
    }
    return neuralNodeOutput();
  }  
  
//==============================================================================
  
  public void saveModel(String modelLoc) {
      
      /** SAVED MODEL FORMAT
       * layer0
       * node0
       * 4 0.8810026686007016
       * 5 0.5564514201581299
       * 6 0.6042151161838017
       * node1
       * 4 0.22580075032373467
       * 5 0.43134952799541526
       * 6 0.13492755106657728
       * Node3----------------------
       * 4 0.23005339489634324
       * 5 0.37658443147084025
       * 6 0.5102098568929565
       */
      
        FileWriter fw = null;
        
        try {
            
            fw = new FileWriter(modelLoc);
            
            int layerCount = neuralNode.length;
            int layerNodes1;
            
            // write the amount of layer
            fw.write(String.valueOf(layerCount) + "\n");
            
            
            // write the amount of nodes for the corresponding layer
            for (int idx0 = 0; idx0 < layerCount; idx0++) {
                
                fw.write(String.valueOf(neuralNode[idx0].length) + "\n");
               
            }
            
           
            for (int idx0 = 0; idx0 < layerCount; idx0++) {
                
                fw.write("layer" + String.valueOf(idx0) + "\n");
                
                layerNodes1 = neuralNode[idx0].length;
                for (int idx1 = 0; idx1 < layerNodes1; idx1++) {
                    
                    fw.write("node" + String.valueOf(neuralNode[idx0][idx1].id) + "\n");
                    
                    for(int key: neuralNode[idx0][idx1].edges.keySet()){
                        //System.out.println("IDs: "+key+" values: "+neuralNode[idx0][idx1].edges.get(key));
                        
                        fw.write(String.valueOf(key));
                        fw.write(" ");
                        fw.write(String.valueOf(neuralNode[idx0][idx1].edges.get(key)));
                        fw.write("\n");
                    }

                }
            }
            
            System.out.println();
            System.out.println("Model SAVED");
            System.out.println();
            
            fw.close();
            
        } catch (Exception e) {
            e.printStackTrace();
        } 
        
  }
  
//==============================================================================
 
    public void readModel(String fileName) {
        
        BufferedReader br = null;
        
        try {

            int totalLayer_read = 0;
            int inputCount_read = 0;
            int hiddenCount_read = 0;
            int outputCount_read = 0;
            String line;
            
            br = new BufferedReader(new FileReader(fileName));
            
            // read the amount of layer
            line = br.readLine();
            totalLayer_read = Integer.parseInt(line);
            
            // read the amount of nodes for each layer
            for (int idx0 = 0; idx0 < totalLayer_read; idx0++) {
                
                line = br.readLine();
                if (idx0 == 0) {
                    inputCount_read = Integer.parseInt(line);
                } else if (idx0 == 1) {
                    
                    if (totalLayer_read == 2) {
                        // no hidden layer
                        outputCount_read = Integer.parseInt(line);
                    } else {
                        // with hidden layer
                        hiddenCount_read = Integer.parseInt(line);
                    }
                    
                } else {
                    outputCount_read = Integer.parseInt(line);
                }
            }
            
            // CONFIRMATION
            System.out.println();
            System.out.println("Confirmation");
            System.out.println("=============================");
            System.out.println("totalLayer: " + totalLayer_read);
            System.out.println("inputCount: " + inputCount_read);
            System.out.println("hiddenCount: " + hiddenCount_read);
            System.out.println("outputCount: " + outputCount_read);
            
            
            int layerCount = 0;  
            neuralNode_read = new Node[totalLayer_read][];

            //input Node
            neuralNode_read[layerCount] = new Node[inputCount_read];
            for(int j=0; j<inputCount_read; j++){
              neuralNode_read[layerCount][j] = new Node();
            }
            layerCount++;

            //hidden Node
            if (totalLayer_read == 2) {
                
                neuralNode_read[layerCount] = new Node[outputCount_read];
                for(int j=0; j<outputCount_read; j++){
                  neuralNode_read[layerCount][j] = new Node();
                }
                
            } else {
                
                neuralNode_read[layerCount] = new Node[hiddenCount_read];
                for(int j=0; j<hiddenCount_read; j++){
                  neuralNode_read[layerCount][j] = new Node();
                }

                layerCount++;
                
                neuralNode_read[layerCount] = new Node[outputCount_read];
                for(int j=0; j<outputCount_read; j++){
                  neuralNode_read[layerCount][j] = new Node();
                }
            }
         
            
            String[] splitted;
            int counter = 0;
            
            for (int idx0 = 0; idx0 < totalLayer_read - 1; idx0++) {
                
                // read "layer"+idx0
                line = br.readLine();
                System.out.println("ok " + line);
                
                // read node
                line = br.readLine();
                System.out.println("ok " + line);
                
                int layerNodes = neuralNode_read[idx0].length;
                for(int j=0; j<layerNodes; j++)
                {
                    counter++;
                    
                    while ((line = br.readLine()) != null) {
                        if (!line.equals("node"+String.valueOf(counter))) {

                            if (line.equals("layer"+String.valueOf(idx0+1))) {
                                break;
                            }
                            
                            // splitted[0] = ID dari next layer
                            // splitted[1] = weight between current node and one node in the next layer
                            splitted = line.split(" ");
                            
                            neuralNode_read[idx0][j].edges.put(Integer.parseInt(splitted[0]), 
                                                            Double.parseDouble(splitted[1]));

                            System.out.println("ok " + splitted[0] + " " + splitted[1]);

                        } else {
                            System.out.println("BREAK: " + line);
                            break;
                        }
                    }
                    
                    if (line.equals("layer"+String.valueOf(idx0+1))) {
                        break;
                    }
                }
                    
            }
            
            br.close();
            
            
        } catch (Exception e) {

            // kasus jika pembacaan file model gagal
            System.out.println("[FAIL] Gagal membaca file model dari: " + fileName + "\n");
            System.out.println("------------------------------------------------------------------------" + "\n");
            e.printStackTrace();
            
        }

    }
 
//==============================================================================
   
  public static void main(String[] args) {
    
    // number of nodes for each layer
    int inputCount_main=1;
    int hiddenCount_main=1;
    int outputCount_main=1;
    int hiddenLayerCount_main=1;
    
    // user input variables
    Scanner scan = new Scanner(System.in);
    String datatrainLoc;
    String datatestLoc;
    String modelLoc;
    
    
    // Build new model or use the existing one
    System.out.println("Pilihan model:");
    System.out.println("0. Build a new model");
    System.out.println("1. Use the existing model");
    System.out.print("Jawaban: ");
    useModel = scan.nextInt();
    scan.nextLine();
    
    
    if (useModel == 0) {
        
        // build a new model
        
        // GET the data train location
        System.out.println("Lokasi data train: ");
        datatrainLoc = "D:\\wekafolder\\data\\student-train.arff";
        
        // GET the amount of hidden layer
        System.out.println("Jumlah hidden layer (max. 1): ");
        hiddenLayerCount_main = scan.nextInt();

        if (hiddenLayerCount_main > 0) {
            
            // GET the amount of nodes in the hidden layer
            System.out.println("Jumlah nodes hidden layer: ");
            hiddenCount_main = scan.nextInt();
            
        }
        
        scan.nextLine();
        
        // GET learning rate
        System.out.println("Learning rate: ");
        learningrate = scan.nextDouble();
        
        scan.nextLine();
        
        // GET the amount of iteration (buildClassifier)
        System.out.println("Number of iteration: ");
        numOfIteration = scan.nextInt();
        
        
        scan.nextLine();
        
        
        // GET the saved model location
        
        modelLoc = "D:\\wekafolder\\model\\newmodel.txt";
        System.out.println("Lokasi penyimpanan model: " + modelLoc);
       
        // Confirmation
        System.out.println("KONFIRMASI");
        System.out.println("===========================");
        System.out.println("Lokasi data train: " + datatrainLoc);
        System.out.println("Lokasi model: " + modelLoc);
        System.out.println("Jumlah hidden layer: " + hiddenLayerCount_main);
        System.out.println("Jumlah nodes hidden layer: " + hiddenCount_main);
        System.out.println("Jumlah iterasi pembentukan model: " + numOfIteration);
        
        
        // READ the dloadata train
        loaddata load = new loaddata(datatrainLoc);
        System.out.println("Banyak atribut adalah " + loaddata.banyakatribut);
        System.out.println("Banyak kelas adalah " + loaddata.banyakkelas);
        
        // Standardisasi data agar range tidak besar sekali
        Instances normalizedDataset = load.train_data;
        
        
        
        try {
			
			//setClassIndex diisikan sesuai dengan index class yang akan dicari
			//untuk team diisikan 12
			//untuk student diisikan 26
			normalizedDataset.setClassIndex(26);
			
			
			
			
			//Bagian ini komentarnya di hilangkan untuk student_train dan Student-mat-test 
			
			//==============================================================================
			Remove m_Filter = new Remove();
			m_Filter.setAttributeIndices("27");
            m_Filter.setInputFormat(normalizedDataset);
			normalizedDataset = Filter.useFilter(normalizedDataset, m_Filter);
			//==============================================================================
			
			normalizedDataset.setClassIndex(26);
			indexClass = normalizedDataset.classIndex();
			Standardization nm = new Standardization(load.train_data);
			normalizedDataset = nm.standardize();

          System.out.println();
          System.out.println("Normalized data train");
          System.out.println(normalizedDataset);
          normalizedDataset.setClassIndex(26);
        } catch (Exception e) {

          e.printStackTrace();

        }
        
        // Print ke layar setelah di standardisasi
        System.out.println(normalizedDataset);
        
        // initialize the amount of layer
        inputCount_main = normalizedDataset.numAttributes()-1;
        outputCount_main = normalizedDataset.numClasses();
        System.out.println("input "+inputCount_main);
        System.out.println("output "+outputCount_main);

        // start the training
        WekaFeed weka = new WekaFeed(inputCount_main, 
                                    hiddenLayerCount_main, 
                                    hiddenCount_main, 
                                    outputCount_main);

        //sebelum FFNN
        //System.out.println("SEBELUM FFNN=====================");
        //weka.printAllEdge();
        //System.out.println("NODE############################");
        //weka.printAllNode();
        //System.out.println("===============================");
        //System.out.println("");


        try {
            
            System.out.println("Building classifier...");
            
            weka.buildClassifier(normalizedDataset);
            
        } catch(Exception e){
            e.printStackTrace();
        }

        
        // SAVE model
        System.out.println("Saving model...");
        weka.saveModel(modelLoc);
        
        
        //setelah FFNN
        System.out.println("SETELAH FFNN=====================");
        System.out.println("NODE############################");
        weka.printAllEdge();
        
        
        
        // EVALUASI DATA TEST 
        
        // Membaca Data test
		
		datatestLoc = "D:\\wekafolder\\data\\student-mat-test.arff";
        loaddata load2 = new loaddata(datatestLoc);
        
        // Standardisasi
        normalizedDataset = load2.train_data;
        
        try {
			
			//setClassIndex diisikan sesuai dengan index class yang akan dicari
			//untuk team diisikan 12
			//untuk student diisikan 26
			normalizedDataset.setClassIndex(26);
			
			
			
			//Bagian ini komentarnya di hilangkan untuk student_train dan Student-mat-test 
			
			//==============================================================================
			Remove m_Filter = new Remove();
			m_Filter.setAttributeIndices("27");
			m_Filter.setInputFormat(normalizedDataset);
			normalizedDataset = Filter.useFilter(normalizedDataset, m_Filter);
			//==============================================================================
			
			normalizedDataset.setClassIndex(26);
			indexClass = normalizedDataset.classIndex();
			Standardization nm = new Standardization(load.train_data);
			normalizedDataset = nm.standardize();

			System.out.println();
			System.out.println("Normalized data train");
			System.out.println(normalizedDataset);
			normalizedDataset.setClassIndex(26);

        } catch (Exception e) {

          e.printStackTrace();

        }
        Instances data1 = normalizedDataset;
        Instances data2 = normalizedDataset;
        
        try{
          
          Evaluation eval = new Evaluation(data1);
          eval.evaluateModel(weka, data2);
          System.out.println(eval.toSummaryString("\nResults ", false));
          System.out.println(eval.toMatrixString());
        }
        catch(Exception e){
          e.printStackTrace();
        }
        
    } else {
        
        // use the existing model
        
        String datatest_demo;
        System.out.println("Lokasi data Team_test: ");
        datatest_demo = scan.nextLine();
        
        
        
        // GET the saved model location
        System.out.println("Lokasi penyimpanan model: ");
        modelLoc = scan.nextLine();
        
        // Confirmation
        System.out.println("KONFIRMASI");
        System.out.println("===========================");
        System.out.println("Lokasi model: " + modelLoc);
        
 
        // READ model
        /*
        WekaFeed weka = new WekaFeed(inputCount_main, 
                                    hiddenLayerCount_main, 
                                    hiddenCount_main, 
                                    outputCount_main);
        */
        
        WekaFeed weka = new WekaFeed();
        
        weka.readModel(modelLoc);
        
        weka.neuralNode = weka.neuralNode_read;
        
        loaddata load = new loaddata(datatest_demo);
        
        // normalize the data train
        Instances normalizedDataset = load.train_data;
        
        try{
          Instances data1 = normalizedDataset;
          Instances data2 = normalizedDataset;
          Evaluation eval = new Evaluation(data1);
          eval.evaluateModel(weka, data2);
          System.out.println(eval.toSummaryString("\nResults ", false));
          System.out.println(eval.toMatrixString());
        }
        catch(Exception e){
          e.printStackTrace();
        }
        // CLASSIFY
        
        
    }

  }

}
