/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.local

import com.salesforce.op.test.TestCommon
import ml.combust.mleap.core.feature._
import ml.combust.mleap.core.types.ScalarShape
import org.apache.spark.ml.linalg.{DenseMatrix, Vectors}
import org.junit.runner.RunWith
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks

@RunWith(classOf[JUnitRunner])
class MLeapModelConverterTest extends PropSpec with PropertyChecks with TestCommon {

  val mleapModels = Table("mleapModels",
    BinarizerModel(0.0, ScalarShape()),
    BucketedRandomProjectionLSHModel(Seq(), 0.0, 0),
    BucketizerModel(Array.empty),
    ChiSqSelectorModel(Seq(), 0),
    CoalesceModel(Seq()),
    CountVectorizerModel(Array.empty, false, 0.0),
    DCTModel(false, 0),
    ElementwiseProductModel(Vectors.zeros(0)),
    FeatureHasherModel(0, Seq(), Seq(), Seq()),
    HashingTermFrequencyModel(),
    IDFModel(Vectors.zeros(0)),
    ImputerModel(0.0, 0.0, ""),
    InteractionModel(Array(), Seq()),
    MathBinaryModel(BinaryOperation.Add),
    MathUnaryModel(UnaryOperation.Log),
    MaxAbsScalerModel(Vectors.zeros(0)),
    MinHashLSHModel(Seq(), 0),
    MinMaxScalerModel(Vectors.zeros(0), Vectors.zeros(0)),
    NGramModel(0),
    NormalizerModel(0.0, 0),
    OneHotEncoderModel(Array()),
    PcaModel(DenseMatrix.zeros(0, 0)),
    PolynomialExpansionModel(0, 0),
    RegexIndexerModel(Seq(), None),
    RegexTokenizerModel(".*".r),
    ReverseStringIndexerModel(Seq()),
    StandardScalerModel(Some(Vectors.dense(Array(1.0))), Some(Vectors.dense(Array(1.0)))),
    StopWordsRemoverModel(Seq(), false),
    StringIndexerModel(Seq()),
    StringMapModel(Map()),
    TokenizerModel(),
    VectorAssemblerModel(Seq()),
    VectorIndexerModel(0, Map()),
    VectorSlicerModel(Array(), Array(), 0),
    WordLengthFilterModel(),
    WordToVectorModel(Map("a" -> 1), Array(1))
  )

  property("convert mleap models to functions") {
    forAll(mleapModels) { m =>
      val fn = MLeapModelConverter.modelToFunction(m)
      fn shouldBe a[Function[_, _]]
    }
  }

  property("error on unsupported models") {
    the[RuntimeException] thrownBy MLeapModelConverter.modelToFunction(model = "not at model") should have message
      "Unsupported MLeap model: java.lang.String"
  }

}
