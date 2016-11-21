
package wekafeed;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Scanner;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.Instance;
import java.lang.*;

public class WekaFeed extends AbstractClassifier{
  
  /* Node attributes */
  public Node[][] neuralNode;
  public static long seed = System.currentTimeMillis();
  public static Random rand = new Random(seed);
  public static int learningrate=1;
  public static int nkelas; // atribut ini ada untuk tes saja
  
  
  /* Used model attributes */
  public static int useModel = 0;
  
  
//==============================================================================
  WekaFeed(int inputCount, int hiddenLayerCount, int hiddenCount, 
          int outputCount){
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
            
            if (useModel == 0) {
             
                // use random values as the weights
                neuralNode[i][j].edges.put(neuralNode[i+1][k].id, rand.nextDouble());
                //neuralNode[i][j].edges.put(neuralNode[i+1][k].id, (double) getidnode(neuralNode[i+1][k])); //untuk bahan tes
            
            } else {
                
                // use the saved values as the weights
                //neuralNode[i][j].edges.put(tbd, tbd);
                
            }
            
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
    
    
    
    for(index=0; index<banyakData; index++)
    {
      //pilih instance
      Instance curr = data.get(index);
      
      
      //create initial value
      //System.out.println("input===============");
      double[] input = new double[banyakAtribut];
      for(int i=0; i<banyakAtribut; i++){
        input[i] = curr.value(i);
        //System.out.println(input[i]);
      }

      //create target
      //System.out.println("target==============");
      System.out.println(curr.classValue());
      double[] target = new double[banyakKelas]; //anggap inisialisasi 0
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
//==============================================================================
  @Override
  public double[] distributionForInstance(Instance instance)
                                 throws java.lang.Exception{
    return neuralNodeOutput();
  }  
//==============================================================================
  
  public void saveModel(String modelLoc) {
      
      /**
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
            
            for (int idx0 = 0; idx0 < layerCount; idx0++) {
                
                fw.write("layer" + String.valueOf(idx0) + "\n");
                
                layerNodes1 = neuralNode[idx0].length;
                for (int idx1 = 0; idx1 < layerNodes1; idx1++) {
                    
                    fw.write("node" + String.valueOf(idx1) + "\n");
                    
                    for(int key: neuralNode[idx0][idx1].edges.keySet()){
                        System.out.println("IDs: "+key+" values: "+neuralNode[idx0][idx1].edges.get(key));
                        
                        fw.write(String.valueOf(key));
                        fw.write(" ");
                        fw.write(String.valueOf(neuralNode[idx0][idx1].edges.get(key)));
                        fw.write("\n");
                    }

                }
            }
            
            System.out.println("Model SAVED");
            
            fw.close();
            
        } catch (Exception e) {
            e.printStackTrace();
        } 
        
  }
  
//==============================================================================
  
  /*
    protected static void readModel(String fileName) {
		
        try {

                // validasi dan load model
                ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
                Object savedModel = in.readObject();
                classifier = (FilteredClassifier) savedModel;
                in.close();

                // output pesan sukses pembacaan file model
                System.out.println("[OK] Membaca file model" + "\n");
                System.out.println("Lokasi file model: " + fileName + "\n");
                System.out.println("------------------------------------------------------------------------" + "\n");

                // menampilkan model yang telah dibaca
                System.out.println(classifier);

        } catch (Exception e) {

                // kasus jika pembacaan file model gagal
                System.out.println("[FAIL] Gagal membaca file model dari: " + fileName + "\n");
                System.out.println("------------------------------------------------------------------------" + "\n");

        }

    }
  */
//==============================================================================
   
  public static void main(String[] args) {
    
    // number of nodes for each layer
    int inputCount=1;
    int hiddenCount=1;
    int outputCount=1;
    
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
    
    
    // GET the data test location
    System.out.println("Lokasi data test: ");
    datatestLoc = scan.nextLine();
    
    
    if (useModel == 0) {
        
        // build a new model
        
        // GET the data train location
        System.out.println("Lokasi data train: ");
        datatrainLoc = scan.nextLine();
        
        // GET the amount of hidden layer
        System.out.println("Jumlah hidden layer (min. 1): ");
        hiddenCount = scan.nextInt();

        scan.nextLine();
        
        // GET learning rate
        System.out.println("Learning rate: ");
        learningrate = scan.nextInt();
        
        scan.nextLine();
        
        // GET the saved model location
        System.out.println("Lokasi penyimpanan model: ");
        modelLoc = scan.nextLine();
        
        // Confirmation
        System.out.println("KONFIRMASI");
        System.out.println("===========================");
        System.out.println("Lokasi data train: " + datatrainLoc);
        System.out.println("Lokasi model: " + modelLoc);
        System.out.println("Jumlah hidden layer: " + hiddenCount);
        
        
        // READ the data train
        loaddata load = new loaddata(datatrainLoc);
        System.out.println("Banyak atribut adalah " + loaddata.banyakatribut);
        System.out.println("Banyak kelas adalah " + loaddata.banyakkelas);

        // initialize the amount of layer
        inputCount = loaddata.banyakatribut;
        outputCount = loaddata.banyakkelas;

        // start the training
        WekaFeed weka = new WekaFeed(inputCount, 1, outputCount, outputCount);

        //sebelum FFNN
        System.out.println("SEBELUM FFNN=====================");
        weka.printAllEdge();
        System.out.println("NODE############################");
        weka.printAllNode();
        System.out.println("===============================");
        System.out.println("");

        try {
            
            System.out.println("Building classifier...");
            weka.buildClassifier(load.train_data);
        
        } catch(Exception e){
            e.printStackTrace();
        }

        
        // SAVE model
        weka.saveModel(modelLoc);
        
        
        //setelah FFNN
        System.out.println("SETELAH FFNN=====================");
        weka.printAllEdge();
        System.out.println("NODE############################");
        weka.printAllNode();
        System.out.println("===============================");
        System.out.println("");
        
    } else {
        
        // use the existing model
        
        // GET the saved model location
        System.out.println("Lokasi penyimpanan model: ");
        modelLoc = scan.nextLine();
        
        // Confirmation
        System.out.println("KONFIRMASI");
        System.out.println("===========================");
        System.out.println("Lokasi model: " + modelLoc);
        
        // READ model
        //readModel(modelLoc);
        
    }
    
    
//    weka.assignInput(new double[]{1,2,3,4});
//    //weka.assignPostEdgeWeight(0, new double[]{4,5,6,7});
//    //weka.assignPreEdgeWeight(5, new double[]{10,11,12,13});
//    //weka.assignEdge(10, 15, 99);
//    System.out.println("Sebelum Dilakukan Feed Forward =>" );
//    System.out.println("Value dan eror masing-masing node");
//    weka.printAllNode();
//    System.out.println();
//    System.out.println("Bobot dari suatu node menuju note dengan ID");
//    weka.printAllEdge();
//    weka.feedforward();
//    System.out.println();
//    System.out.println("Setelah Dilakukan Feed Forward =>" );
//    System.out.println();
//    System.out.println("Value dan eror masing-masing node");
//    weka.printAllNode();
//    System.out.println();
//    weka.backpropagation(new double[]{1,0,0}); // Hanya memiliki satu node di layer output
//    System.out.println("Setelah dilakukan Back Propagation =>" );
//    System.out.println();
//    System.out.println("Value dan eror masing-masing node");
//    weka.printAllNode();
//    System.out.println();
//    System.out.println("Bobot dari suatu node menuju note dengan ID");
//    weka.printAllEdge();
//    
//   
//    // convert nominal to numeric for class
//    
//
//    
//    // dataset preprocessing
//    /*try {
//    
//        Normalization nm = new Normalization(load.train_data);
//        Instances normalizedDataset = nm.normalize();
//    
//        System.out.println();
//       // System.out.println("Normalized data train");
//       // System.out.println(normalizedDataset);
//        
//    } catch (Exception e) {
//        
//        e.printStackTrace();
//       
//    }
//    */
  }

}
