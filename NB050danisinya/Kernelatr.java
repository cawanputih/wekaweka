/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucil;

import weka.core.Statistics;
import weka.core.Utils;

/**
 *
 * @author mac
 */
public class Kernelatr extends Typeatr {
    double precision;
    double[] values;
    double[] weights;
    int numvalue;
    double sumweight;
    boolean boolweight;
    double stdev;
    static double maxerror = 0.01;
    
    private double round(double data) 
    {
        return Math.rint(data / precision) * precision;
    }
    
    public Kernelatr(double pres)
    {
        values = new double [50];
        weights = new double [50];
        numvalue = 0;
        sumweight = 0;
        boolweight = true;
        precision = pres;
        if (precision < Utils.SMALL) precision = Utils.SMALL;
        stdev = precision / (6);
    }

    private int findNearestValue(double key) {
        int low = 0; 
        int high = numvalue;
        int middle;
        while (low < high) {
          middle = (low + high) / 2;
          double current = values[middle];
          if (current == key) {
            return middle;
          }
          if (current > key) {
            high = middle;
          } else if (current < key) {
            low = middle + 1;
          }
        }
        return low;
      }
    
    
    @Override
    public double getProbability(int d) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void addValue(double data, double weight) {
        if (weight == 0) {
      return;
    }
    data = round(data);
    int insertIndex = findNearestValue(data);
    if ((numvalue <= insertIndex) || (values[insertIndex] != data)) {
      if (numvalue < values.length) {
        int left = (numvalue - insertIndex); 
        System.arraycopy(values, insertIndex, 
            values, insertIndex + 1, left);
        System.arraycopy(weights, insertIndex, 
            weights, insertIndex + 1, left);
        
        values[insertIndex] = data;
        weights[insertIndex] = weight;
        numvalue++;
      } else {
        double [] newvalues = new double [values.length * 2];
        double [] newweights = new double [values.length * 2];
        int left = numvalue - insertIndex; 
        System.arraycopy(values, 0, newvalues, 0, insertIndex);
        System.arraycopy(weights, 0, newweights, 0, insertIndex);
        newvalues[insertIndex] = data;
        newweights[insertIndex] = weight;
        System.arraycopy(values, insertIndex, 
            newvalues, insertIndex + 1, left);
        System.arraycopy(weights, insertIndex, 
            newweights, insertIndex + 1, left);
        numvalue++;
        values = newvalues;
        weights = newweights;
      }
      if (weight != 1) {
        boolweight = false;
      }
    } else {
      weights[insertIndex] += weight;
      boolweight = false;      
    }
    sumweight += weight;
    double range = values[numvalue - 1] - values[0];
    if (range > 0) {
      stdev = Math.max(range / Math.sqrt(sumweight), 
          precision / (6));
    }
    }

    @Override
    public double getProbability(double data) {
    double delta , sum = 0, currentProb;
    double zLower, zUpper;
    if (numvalue == 0) {
      zLower = (data - (precision / 2)) / stdev;
      zUpper = (data + (precision / 2)) / stdev;
      return (Statistics.normalProbability(zUpper)
	      - Statistics.normalProbability(zLower));
    }
    
    double weightSum = 0;
    int start = findNearestValue(data);    
    for (int i = start; i < numvalue; i++) {
      delta = values[i] - data;
      zLower = (delta - (precision / 2)) / stdev;
      zUpper = (delta + (precision / 2)) / stdev;
      currentProb = (Statistics.normalProbability(zUpper) - Statistics.normalProbability(zLower));
      sum += currentProb * weights[i];
      
      weightSum += weights[i];
      if (currentProb * (sumweight - weightSum) < sum * maxerror) {
	break;
      }
    }
    for (int i = start - 1; i >= 0; i--) {
      delta = values[i] - data;
      zLower = (delta - (precision / 2)) / stdev;
      zUpper = (delta + (precision / 2)) / stdev;
      currentProb = (Statistics.normalProbability(zUpper)- Statistics.normalProbability(zLower));
      sum += currentProb * weights[i];
      weightSum += weights[i];
      if (currentProb * (sumweight - weightSum) < sum * maxerror) {
	break;
      }
    }
    return sum / sumweight;
    }
    
}
