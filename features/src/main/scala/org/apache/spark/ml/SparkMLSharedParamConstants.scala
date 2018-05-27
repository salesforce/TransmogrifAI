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

package org.apache.spark.ml

import com.salesforce.op.UID
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasLabelCol, _}

/**
 * Class containing constants used by spark ML for common parameters (mostly column names).
 */
case object SparkMLSharedParamConstants
  extends HasLabelCol
    with HasFeaturesCol
    with HasPredictionCol
    with HasRawPredictionCol
    with HasProbabilityCol
    with HasInputCol
    with HasOutputCol {
  // These are the typical names for the parameters in the sparkML algorithms:
  // target variable
  val LabelColName = labelCol.name
  // feature vector
  val FeaturesColName = featuresCol.name
  // response variable
  val PredictionColName = predictionCol.name

  // these are the names for input/ouput for estimators/transformers that transform a single column to another
  // single columns
  val InputColName = inputCol.name
  val OutputColName = outputCol.name

  // These param names below are used in probabilistic classifiers in sparkML, where 'raw' simply
  // represents an unnormalized version of the other.
  val RawPredictionColName = rawPredictionCol.name
  val ProbabilityColName = probabilityCol.name

  // ********************************************************************************
  // just adding these below so I can have this object (static class) for constants
  override def copy(extra: ParamMap): SparkMLSharedParamConstants.type = copyValues(this, extra)
  override val uid: String = UID(this.getClass)
  // ********************************************************************************

  type InOutTransformer = Transformer with HasInputCol with HasOutputCol
}


