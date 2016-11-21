/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucil;

import weka.core.Statistics;

/**
 *
 * @author mac
 */
public class Normalatr extends Typeatr{
    double precision;
    double stdev;
    double sumweight;
    double sumvalue;
    double sumvaluesq = Math.sqrt(sumvalue);
    double mean;
    
    Normalatr(double pres)
    {
        precision =pres;
        stdev = precision / (6);
    }
    
    private double round(double data) {

        return Math.rint(data / precision) * precision;
    }
    
    @Override
    public void addValue(double data, double weight) {

        if (weight == 0) {
          return;
        }
        data = round(data);
        sumweight += weight;
        sumweight += data * weight;
        sumvaluesq += data * data * weight;

        if (sumweight > 0) {
          mean = sumvalue / sumweight;
          double stdDev = Math.sqrt(Math.abs(sumvaluesq - mean * sumvalue)/ sumweight);
          if (stdDev > 1e-10) {
            stdev = Math.max(precision / (6),stdDev);
          }
        }
    }
    
    
    @Override
    public double getProbability(double data) {
        data = round(data);
        double zlower = (data - mean - (precision / 2)) / stdev;
        double zupper = (data - mean + (precision / 2)) / stdev;

        double plower = Statistics.normalProbability(zlower);
        double pupper = Statistics.normalProbability(zupper);
        return pupper - plower;
  }

    @Override
    public double getProbability(int d) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
