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
import org.apache.spark.ml.linalg.Vector

/**
 * Converter of MLeap model instances to a model apply function
 */
case object MLeapModelConverter {

  /**
   * Convert MLeap model instance to a model apply function
   * TODO look into using mleap transform function instead of model apply
   * @param model MLeap model
   * @throws RuntimeException if model type is not supported
   * @return runnable model apply function
   */
  def modelToFunction(model: Any): Array[Any] => Any = model match {
    case m: Binarizer => x => m.model.apply(x(0).asInstanceOf[Number].doubleValue())
    case m: BucketedRandomProjectionLSH => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: Bucketizer => x => m.model.apply(x(0).asInstanceOf[Number].doubleValue())
    case m: ChiSqSelector => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: Coalesce => x => m.model.apply(x: _*)
    case m: CountVectorizer => x => m.model.apply(x(0).asInstanceOf[Seq[String]])
    case m: DCT => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: ElementwiseProduct => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: FeatureHasher => x => m.model.apply(x(0).asInstanceOf[Seq[Any]])
    case m: HashingTermFrequency => x => m.model.apply(x(0).asInstanceOf[Seq[Any]])
    case m: IDF => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: Imputer => x => m.model.apply(x(0).asInstanceOf[Number].doubleValue())
    case m: Interaction => x => m.model.apply(x(0).asInstanceOf[Seq[Any]])
    case m: MathBinary => x => m.model.apply(
        x.headOption.map(_.asInstanceOf[Number].doubleValue()),
        x.lastOption.map(_.asInstanceOf[Number].doubleValue())
      )
    case m: MathUnary => x => m.model.apply(x(0).asInstanceOf[Number].doubleValue())
    case m: MaxAbsScaler => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: MinHashLSH => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: MinMaxScaler => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: NGram => x => m.model.apply(x(0).asInstanceOf[Seq[String]])
    case m: Normalizer => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: OneHotEncoder => x => m.model.apply(x(0).asInstanceOf[Vector].toArray)
    case m: Pca => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: PolynomialExpansion => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: RegexIndexer => x => m.model.apply(x(0).toString)
    case m: RegexTokenizer => x => m.model.apply(x(0).toString)
    case m: ReverseStringIndexer => x => m.model.apply(x(0).asInstanceOf[Number].intValue())
    case m: StandardScaler => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: StopWordsRemover => x => m.model.apply(x(0).asInstanceOf[Seq[String]])
    case m: StringIndexer => x => m.model.apply(x(0))
    case m: StringMap => x => m.model.apply(x(0).toString)
    case m: Tokenizer => x => m.model.apply(x(0).toString)
    case m: VectorAssembler => x => m.model.apply(x(0).asInstanceOf[Seq[Any]])
    case m: VectorIndexer => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: VectorSlicer => x => m.model.apply(x(0).asInstanceOf[Vector])
    case m: WordLengthFilter => x => m.model.apply(x(0).asInstanceOf[Seq[String]])
    case m: WordToVector => x => m.model.apply(x(0).asInstanceOf[Seq[String]])
    case m => throw new RuntimeException(s"Unsupported MLeap model: ${m.getClass.getName}")
  }

}
