/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
