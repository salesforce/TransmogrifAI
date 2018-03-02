/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

// scalastyle:off
// TODO(vlad): make sure that a simple intellij run fills in the resources
// @see https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/util/MetadataUtils.scala#L54
// scalastyle:on
import java.io.File

import com.salesforce.op.cli.gen.Ops
import org.apache.commons.io.FileUtils

class CliExec {
  protected val DEBUG = false

  private[cli] def delete(dir: File): Unit = {
    FileUtils.deleteDirectory(dir)
    if (dir.exists()) {
      throw new IllegalStateException(s"Directory '${dir.getAbsolutePath}' still exists")
    }
  }

  def main(args: Array[String]): Unit = try {
    val outcome = for {
      arguments <- CommandParser.parse(args, CliParameters())
      if arguments.command == "gen"
      settings <- arguments.values
    } yield Ops(settings).run()

    outcome getOrElse {
      CommandParser.showUsage()
      quit("wrong arguments", 1)
    }
  } catch {
    case x: Exception =>
      if (DEBUG) x.printStackTrace()
      val msg = Option(x.getMessage).getOrElse(x.getStackTrace.mkString("", "\n", "\n"))
      quit(msg)
  }

  def quit(errorMsg: String, code: Int = -1): Nothing = {
    System.err.println(errorMsg)
    sys.exit(code)
  }
}

object CLI {
  def main(args: Array[String]): Unit = (new CliExec).main(args)
}
