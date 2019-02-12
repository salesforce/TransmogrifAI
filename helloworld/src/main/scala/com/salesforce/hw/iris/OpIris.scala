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

package com.salesforce.hw.iris

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.readers.CustomReader
import com.salesforce.op.stages.impl.classification.MultiClassificationModelSelector
import com.salesforce.op.stages.impl.tuning.DataCutter
import com.salesforce.op.utils.kryo.OpKryoRegistrator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

/**
 * TransmogrifAI MultiClass Classification example on the Iris Dataset
 */
object OpIris extends OpAppWithRunner with IrisFeatures {

  override def kryoRegistrator: Class[_ <: OpKryoRegistrator] = classOf[IrisKryoRegistrator]

  ////////////////////////////////////////////////////////////////////////////////
  // READER DEFINITIONS
  /////////////////////////////////////////////////////////////////////////////////

  val randomSeed = 42

  val irisReader = new CustomReader[Iris](key = _.getID.toString){
    def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Iris], Dataset[Iris]] = {
      val path = getFinalReadPath(params)
      val myFile = spark.sparkContext.textFile(path)

      Left(myFile.filter(_.nonEmpty).zipWithIndex.map { case (x, number) =>
        val words = x.split(",")
        new Iris(number.toInt, words(0).toDouble, words(1).toDouble, words(2).toDouble, words(3).toDouble, words(4))
      })
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // WORKFLOW DEFINITION
  /////////////////////////////////////////////////////////////////////////////////

  val labels = irisClass.indexed()

  val features = Seq(sepalLength, sepalWidth, petalLength, petalWidth).transmogrify()

  val cutter = DataCutter(reserveTestFraction = 0.2, seed = randomSeed)

  val prediction = MultiClassificationModelSelector
    .withCrossValidation(splitter = Option(cutter), seed = randomSeed)
    .setInput(labels, features).getOutput()

  val evaluator = Evaluators.MultiClassification.f1().setLabelCol(labels).setPredictionCol(prediction)

  val workflow = new OpWorkflow().setResultFeatures(prediction, labels)

  def runner(opParams: OpParams): OpWorkflowRunner =
    new OpWorkflowRunner(
      workflow = workflow,
      trainingReader = irisReader,
      scoringReader = irisReader,
      evaluationReader = Option(irisReader),
      evaluator = Option(evaluator),
      featureToComputeUpTo = Option(features)
    )
}
