/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen.templates

// scalastyle:off
class SampleObject {
  private def ignore = throw new NotImplementedError("it's just a prototype")
  def codeGeneration_categoricalField_codeGeneration[E <: Enum[E]] : E = ignore
  def codeGeneration_textField_codeGeneration : String = ignore
  def codeGeneration_realField_codeGeneration : Double = ignore
  def codeGeneration_binaryField_codeGeneration : Boolean = ignore
  def codeGeneration_integralField_codeGeneration : Integer = ignore

  // the following lines serve the purpose of ensuring that our code will compile
  private val br = new BinaryFeatureTemplate().feature.asResponse
  private val bp = new BinaryFeatureTemplate().feature.asPredictor
  private val cr = new CategoricalFeatureTemplate().feature.asResponse
  private val cp = new CategoricalFeatureTemplate().feature.asPredictor
  private val ir = new IntegralFeatureTemplate().feature.asResponse
  private val ip = new IntegralFeatureTemplate().feature.asPredictor
  private val rr = new RealFeatureTemplate().feature.asResponse
  private val rp = new RealFeatureTemplate().feature.asPredictor
  private val tr = new TextFeatureTemplate().feature.asResponse
  private val tp = new TextFeatureTemplate().feature.asPredictor
}
