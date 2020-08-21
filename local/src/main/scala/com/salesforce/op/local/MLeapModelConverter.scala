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

import ml.combust.mleap.runtime.transformer.feature._
import ml.combust.mleap.core.feature._
import org.apache.spark.ml.linalg.Vector

/**
 * Converter of MLeap model instances to a model apply function
 */
case object MLeapModelConverter {

  /**
   * Convert MLeap model instance to a model apply function
   * @param model MLeap model
   * @throws RuntimeException if model type is not supported
   * @return runnable model apply function
   */
  def modelToFunction(model: Any): Array[Any] => Any = model match {
    // TODO look into applying transform directly rather than using the model apply
    case m: BinarizerModel => x => m.apply(x(0).asInstanceOf[Number].doubleValue())
    case m: BucketedRandomProjectionLSHModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: BucketizerModel => x => m.apply(x(0).asInstanceOf[Number].doubleValue())
    case m: ChiSqSelectorModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: CoalesceModel => x => m.apply(x: _*)
    case m: CountVectorizerModel => x => m.apply(x(0).asInstanceOf[Seq[String]])
    case m: DCTModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: ElementwiseProductModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: FeatureHasherModel => x => m.apply(x(0).asInstanceOf[Seq[Any]])
    case m: HashingTermFrequencyModel => x => m.apply(x(0).asInstanceOf[Seq[Any]])
    case m: IDFModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: ImputerModel => x => m.apply(x(0).asInstanceOf[Number].doubleValue())
    case m: InteractionModel => x => m.apply(x(0).asInstanceOf[Seq[Any]])
    case m: MathBinaryModel => x =>
      m.apply(
        x.headOption.map(_.asInstanceOf[Number].doubleValue()),
        x.lastOption.map(_.asInstanceOf[Number].doubleValue())
      )
    case m: MathUnaryModel => x => m.apply(x(0).asInstanceOf[Number].doubleValue())
    case m: MaxAbsScalerModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: MinHashLSHModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: MinMaxScalerModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: NGramModel => x => m.apply(x(0).asInstanceOf[Seq[String]])
    case m: NormalizerModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: OneHotEncoderModel => x => m.apply(x(0).asInstanceOf[Vector].toArray)
    case m: PcaModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: PolynomialExpansionModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: RegexIndexerModel => x => m.apply(x(0).toString)
    case m: RegexTokenizerModel => x => m.apply(x(0).toString)
    case m: ReverseStringIndexerModel => x => m.apply(x(0).asInstanceOf[Number].intValue())
    case m: StandardScalerModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: StopWordsRemoverModel => x => m.apply(x(0).asInstanceOf[Seq[String]])
    case m: StringIndexerModel => x => m.apply(x(0))
    case m: StringMapModel => x => m.apply(x(0).toString)
    case m: TokenizerModel => x => m.apply(x(0).toString)
    case m: VectorAssemblerModel => x => m.apply(x(0).asInstanceOf[Seq[Any]])
    case m: VectorIndexerModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: VectorSlicerModel => x => m.apply(x(0).asInstanceOf[Vector])
    case m: WordLengthFilterModel => x => m.apply(x(0).asInstanceOf[Seq[String]])
    case m: WordToVectorModel => x => m.apply(x(0).asInstanceOf[Seq[String]])
    case m => throw new RuntimeException(s"Unsupported MLeap model: ${m.getClass.getName}")
  }

}
