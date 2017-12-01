/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

// scalastyle:off
// TODO(vlad): make sure that a simple intellij run fills in the resources
// @see https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/util/MetadataUtils.scala#L54
// scalastyle:on
import com.salesforce.op.cli.gen.Ops

object CLI {

  def main(args: Array[String]): Unit = try {
    val outcome = for {
      arguments <- CommandParser.parse(args, CliParameters())
      if arguments.command == "gen"
      settings <- arguments.settings.values
    } yield Ops(settings).run()

    outcome getOrElse {
      CommandParser.showUsage()
      sys.exit(1)
    }
  } catch {
    case x: Exception => quit(x.getMessage)
  }

  def quit(errorMsg: String): Nothing = {
    System.err.println(errorMsg)
    sys.exit(-1)
  }

}
