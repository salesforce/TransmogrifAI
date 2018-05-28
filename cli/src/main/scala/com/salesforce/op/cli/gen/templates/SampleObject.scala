/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
