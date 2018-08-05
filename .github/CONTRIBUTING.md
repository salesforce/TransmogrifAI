# Contributing to TransmogrifAI

This page lists recommendations and requirements for how to best contribute to TransmogrifAI. We strive to obey these as best as possible. As always, thanks for contributing – we hope these guidelines make it easier and shed some light on our approach and processes.

# Issues, requests & ideas

Use GitHub [Issues](https://github.com/salesforce/TransmogrifAI/issues) page to submit issues, enhancement requests and discuss ideas.

# Contributing

1. **Ensure the bug/feature was not already reported** by searching on GitHub under [Issues](https://github.com/salesforce/TransmogrifAI/issues).  If none exists, create a new issue so that other contributors can keep track of what you are trying to add/fix and offer suggestions (or let you know if there is already an effort in progress).
3. **Clone** the forked repo to your machine.
4. **Commit** changes to your own branch.
5. **Push** your work back up to your fork.
6. **Submit** a [Pull Request](https://github.com/salesforce/TransmogrifAI/pulls) against the `master` branch and refer to the issue(s) you are fixing. Try not to pollute your pull request with unintended changes. Keep it simple and small.

> **NOTE**: Be sure to [sync your fork](https://help.github.com/articles/syncing-a-fork/) before making a pull request.

# Contribution Checklist

- [x] Clean, simple, well styled code
- [x] Comments
  - Module-level & function-level comments.
  - Comments on complex blocks of code or algorithms (include references to sources).
- [x] Tests
  - Increase code coverage, not versa.
  - Use [ScalaTest](http://www.scalatest.org/) with `FlatSpec` and `PropSpec`.
  - Use our testkit that contains a bunch of testing facilities you would need. Simply `import com.salesforce.op.test._` and borrow inspiration from existing tests.
- [x] Dependencies
  - Minimize number of dependencies.
  - Prefer BSD, Apache 2.0, MIT, ISC and MPL licenses.

# Code of Conduct
Follow the [Apache Code of Conduct](https://www.apache.org/foundation/policies/conduct.html).

# License
By contributing your code, you agree to license your contribution under the terms of the [BSD 3-Clause](License).
