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

import com.salesforce.op.{OpParams, OpWorkflow}
import org.apache.spark.sql.SparkSession


/**
 * A class for running TransmogrifAI Workflow without Spark.
 *
 * @param workflow the workflow that you want to run (Note: the workflow should have the resultFeatures set)
 */
class OpWorkflowRunnerLocal(val workflow: OpWorkflow) extends Serializable {

  /**
   * Load the model & prepare a score function for local scoring
   *
   * Note: since we use Spark native [[org.apache.spark.ml.util.MLWriter]] interface
   * to load stages the Spark session is being created internally. So if you would not like
   * to have an open SparkSession please make sure to stop it after creating the score function:
   *
   *   val scoreFunction = new OpWorkflowRunnerLocal(workflow).score(params)
   *   // stop the session after creating the scoreFunction if needed
   *   SparkSession.builder().getOrCreate().stop()
   *
   * @param params params to use during scoring
   * @param spark  spark session needed for preparing scoring function.
   *               Once scoring function is returned the session then can be shutdown as it's not used during scoring
   * @return score function for local scoring
   */
  def score(params: OpParams)(implicit spark: SparkSession): ScoreFunction = {
    require(params.modelLocation.isDefined, "Model location must be set in params")
    val model = workflow.loadModel(params.modelLocation.get)
    model.scoreFunction
  }

}
