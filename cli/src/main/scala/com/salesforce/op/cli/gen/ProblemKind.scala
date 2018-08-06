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

package com.salesforce.op.cli.gen

import com.salesforce.op.cli.SchemaSource

/**
 * Represents one of the three kinds of machine learning problems.
 */
sealed trait ProblemKind

object ProblemKind {

  case object BinaryClassification extends ProblemKind
  case object MultiClassification extends ProblemKind
  case object Regression extends ProblemKind

  /**
   * Holds every possible [[ProblemKind]]
   */
  val values: List[ProblemKind] = List(BinaryClassification, MultiClassification, Regression)

  /**
   * Ask the user to identify the problem kind (because we were unable to identify it)
   *
   * @param responseFieldSchema The avro field for the response field, we display the name to the user
   * @param available What [[ProblemKind]]s are available for us to prompt the user with
   * @return The [[ProblemKind]] the user picked
   */
  private[gen] def askKind(
    ops: Ops,
    responseFieldSchema: AvroField,
    available: List[ProblemKind] = values): ProblemKind = {

    val options = Map(
      Regression -> List("regress", "regression"),
      BinaryClassification -> List("binclass", "binary classification"),
      MultiClassification -> List("multiclass", "multi classification")
    ).filterKeys(available.contains)

    ops.ask(
      "Cannot infer the kind of problem based on response field" +
        s" '${responseFieldSchema.name}'. What kind of problem is this?",
      options,
      s"Failed to figure out problem kind from $responseFieldSchema")
  }
}
