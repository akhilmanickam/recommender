package com.example.controller;

import java.io.IOException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Test {

	
	/**
	 * @param args
	 * @throws IOException 
	 * @throws InvalidKerasConfigurationException 
	 * @throws UnsupportedKerasConfigurationException 
	 */
	public static void main(String[] args) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
		// TODO Auto-generated method stub
		//Connection con =  DBConnection.getConnection();
		Test.loadPredictBooks(); 

	}

	private static void loadPredictBooks() {
		int uid = 12;
		Connection con;
		Statement stmt;
		ResultSet rs;
		if (uid != 0) {
			con = DBConnection.getConnection();

			try {
				stmt = con.createStatement();
				rs = stmt
						.executeQuery("select * from user_transaction where uid ="
								+ uid);

				INDArray indBooks =  Nd4j.zeros(10000);
				
				for (int i = 0; i < 10000; i++) {
					indBooks.putScalar(new int[]{i}, i+1);
				}

				INDArray userBooks =  Nd4j.zeros(10000);

				while (rs.next()) {
					// String name = rs.getString("name");
					int id = rs.getInt("uid");
					int bookId = rs.getInt("bookid");
					userBooks.putScalar((new int[] {bookId-1}), 1);
				}
				String fullModel = new ClassPathResource("regression_model2.h5").getFile().getPath();		
				ComputationGraph model = KerasModelImport.importKerasModelAndWeights(fullModel);
				model.init();
				INDArray[] prediction;
				prediction = model.output(userBooks,indBooks);
				System.out.println(prediction);
						

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
}
