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

package com.salesforce.op.stages.impl.evaluator

import com.salesforce.op.evaluators.{Evaluators, OpBinaryClassificationEvaluatorBase, OpMultiClassificationEvaluatorBase, SingleMetric}
import com.twitter.algebird.AveragedValue
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset
import com.salesforce.op.utils.spark.RichDataset.RichDataset

/**
 * Logarithmic Loss metric, implemented as both Binary and MultiClass evaluators
 */
object LogLoss {

  private def logLossFun(ds: Dataset[(Double, Vector, Vector, Double)]): Double = {
    import ds.sparkSession.implicits._
    require(!ds.isEmpty, "Dataset is empty, log loss cannot be calculated")
    val avg = ds.map { case (lbl, _, prob, _) =>
      new AveragedValue(count = 1L, value = -math.log(prob.toArray(lbl.toInt)))
    }.reduce(_ + _)
    avg.value
  }

  def binaryLogLoss: OpBinaryClassificationEvaluatorBase[SingleMetric] = Evaluators.BinaryClassification.custom(
    metricName = "BinarylogLoss",
    largerBetter = false,
    evaluateFn = logLossFun
  )

  def multiLogLoss: OpMultiClassificationEvaluatorBase[SingleMetric] = Evaluators.MultiClassification.custom(
    metricName = "MultiClasslogLoss",
    largerBetter = false,
    evaluateFn = logLossFun
  )
}
