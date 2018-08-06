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

package com.salesforce.hw.titanic

import com.salesforce.op._
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry._
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.tuning.DataSplitter
import com.salesforce.op.utils.kryo.OpKryoRegistrator

/**
 * TransmogrifAI example classification app using the Titanic dataset
 */
object OpTitanic extends OpAppWithRunner with TitanicFeatures {

  ////////////////////////////////////////////////////////////////////////////////
  // READER DEFINITION
  /////////////////////////////////////////////////////////////////////////////////

  val randomSeed = 112233
  val simpleReader = DataReaders.Simple.csv[Passenger](
    schema = Passenger.getClassSchema.toString, key = _.getPassengerId.toString
  )

  ////////////////////////////////////////////////////////////////////////////////
  // WORKFLOW DEFINITION
  /////////////////////////////////////////////////////////////////////////////////

  val featureVector = Seq(pClass, name, sex, age, sibSp, parch, ticket, cabin, embarked).transmogrify()

  val checkedFeatures = survived.sanityCheck(
    featureVector = featureVector, checkSample = 1.0, sampleSeed = randomSeed, removeBadFeatures = true
  )

  val splitter = DataSplitter(seed = randomSeed, reserveTestFraction = 0.1)

  val (pred, raw, prob) = BinaryClassificationModelSelector
    .withCrossValidation(splitter = Option(splitter), seed = randomSeed)
    .setLogisticRegressionRegParam(0.05, 0.1)
    .setLogisticRegressionElasticNetParam(0.01)
    .setRandomForestMaxDepth(5, 10)
    .setRandomForestMinInstancesPerNode(10, 20, 30)
    .setRandomForestSeed(randomSeed)
    .setModelsToTry(LogisticRegression, RandomForest)
    .setInput(survived, checkedFeatures)
    .getOutput()

  val workflow = new OpWorkflow().setResultFeatures(pred, raw)

  val evaluator = Evaluators.BinaryClassification.auPR()
    .setLabelCol(survived)
    .setPredictionCol(pred)
    .setRawPredictionCol(raw)

  ////////////////////////////////////////////////////////////////////////////////
  // APPLICATION RUNNER DEFINITION
  /////////////////////////////////////////////////////////////////////////////////
  def runner(opParams: OpParams): OpWorkflowRunner =
    new OpWorkflowRunner(
      workflow = workflow,
      trainingReader = simpleReader,
      scoringReader = simpleReader,
      evaluationReader = Option(simpleReader),
      evaluator = Option(evaluator),
      featureToComputeUpTo = Option(featureVector)
    )

  override def kryoRegistrator: Class[_ <: OpKryoRegistrator] = classOf[TitanicKryoRegistrator]

}
