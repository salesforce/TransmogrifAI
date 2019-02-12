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

package com.salesforce.op.utils.spark


import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.sql.Dataset
import org.slf4j.LoggerFactory

/**
 * Various [[Evaluator]] helpers functions
 */
case object RichEvaluator {

  import com.salesforce.op.utils.spark.RichDataset._

  private val log = LoggerFactory.getLogger(getClass.getName.stripSuffix("$"))

  /**
   * Various [[Evaluator]] helpers functions
   */
  implicit class RichEvaluator(val evaluator: Evaluator) extends AnyVal {

    /**
     * Safely evaluates model output and returns a scalar metric only if the dataset is not empty,
     * otherwise returns the default metric value.
     *
     * @param dataset a dataset that contains labels/observations and predictions.
     * @param default default metric value to return if dataset is empty
     * @return evaluated metric or default
     */
    def evaluateOrDefault(dataset: Dataset[_], default: => Double): Double = {
      if (dataset.isEmpty) {
        val defaultValue = default
        log.warn("The dataset is empty. Returning default metric value: {}.", defaultValue)
        defaultValue
      }
      else evaluator.evaluate(dataset)
    }

  }

}
