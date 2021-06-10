# Geometric Deep Learning on Diffusion MRI

The dataset was kindly provided by [Centre for the Developing Brain](http://cmic.cs.ucl.ac.uk/cdmri20/challenge.html).

## Running the notebooks

### Environment variables and .env file

Create a `.env` file in the root directory of this project.
`utils.env.py` will load file and set the variables.
If the file is not present it will use the system environment variables.

The following environment variables can be declared:

<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>DATA_PATH</code></td>
            <td>
                The path to the MUDI data. The directory structure is expected to be the same as provided by
                <a href="http://cmic.cs.ucl.ac.uk/cdmri20/challenge.html">Centre for the Developing Brain</a>.
            </td>
        </tr>
        <tr>
            <td><code>LOGGING_LEVEL</code></td>
            <td>
                <b>Optional.</b>
                Default value is <code>30</code>.
                It is recommend to set this <code>&gt;20</code> when training the models for real.
                <table>
                    <thead>
                        <tr>
                            <th>Level</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>CRITICAL</td>
                            <td>50</td>
                        </tr>
                        <tr>
                            <td>ERROR</td>
                            <td>40</td>
                        </tr>
                        <tr>
                            <td>WARNING</td>
                            <td>30</td>
                        </tr>
                        <tr>
                            <td>INFO</td>
                            <td>20</td>
                        </tr>
                        <tr>
                            <td>DEBUG</td>
                            <td>10</td>
                        </tr>
                        <tr>
                            <td>NOTSET</td>
                            <td>0</td>
                        </tr>
                    </tbody>
                </table>
            </td>
        </tr>
    </tbody>
</table>

Example:

```
DATA_PATH=/home/user/MUDI
LOGGING_LEVEL=10
```
