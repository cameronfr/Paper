// Cameron Franz, September 2018.
// TODO: novelty, 200ms wait before search to reduce lag, aesthetics, cleaning up word results.

const vectorURL = "https://storage.googleapis.com/mlstorage-cloud/Data/glove.6B.50d.txt.zip"

//fetch doesn't have progress, file too big to not have indicator ==> xhr ugliness.
function fetchWordVectors(onProgress) {
  return new Promise((resolve, reject) => {
    localforage.setDriver(localforage.INDEXEDDB)
    localforage.getItem('vectorZip').then(zip => {
      if (zip == null) {
        var xhr = new XMLHttpRequest();
        xhr.open("GET", vectorURL, true);
        xhr.responseType = "arraybuffer";
        xhr.onerror = (e) => {
          console.log("✗ Error fetching vector file:\n" + xhr.statusText)
          reject(xhr.statusText)
        }
        xhr.onload = (e) => {
          let zip = xhr.response
          localforage.setItem('vectorZip', zip).then(() => {
            console.log("✓ Downloaded and stored vector file")
            resolve(zip)
          })
        }
        xhr.onprogress = (e) => {
          onProgress(e.loaded / e.total)
        }
        xhr.send();
      }
      else {
        console.log("✓ Found stored vector file")
        resolve(zip)
      }
    })
  })
}

async function loadWordVectors(zipArrayBuffer) {
  let text = await JSZip.loadAsync(zipArrayBuffer).then(res => res.files["glove.6B.50d.txt"].async("string"))
    .catch(err => console.log("✗ Error unzipping vector file:\n" + err))

  let lines = text.split("\n")
  lines = lines.slice(0, -1)

  let wordDict = {fromIdx: {}, fromWord: {}}
  let wordVectors = tf.buffer([lines.length, 50], "float32")

  for (var idx in lines) {
    let line = lines[idx]
    let sep = line.indexOf(" ")
    let word = line.substring(0, sep)
    let vector = line.substring(sep+1).split(' ')
    vector = vector.map(x => parseFloat(x))

    wordDict.fromIdx[parseInt(idx)] = word
    wordDict.fromWord[word] = idx

    let magnitude = Math.sqrt(vector.reduce((acc, x) => acc + x*x))

    for (var jdx in vector) {
      let feat = vector[jdx] / magnitude
      wordVectors.set(feat, parseInt(idx), parseInt(jdx))
    }
  }

  console.log("✓ Loaded word vector matrix of shape " + wordVectors.shape)
  wordVectors = wordVectors.toTensor()
  return [wordVectors, wordDict]

}

function getHighestKIndices(array, k) {
  let highestValues = Array(k)
  for (var i=0; i<array.length; i++) {
    for (var j=0; j<highestValues.length; j++) {
      if (highestValues[j] == undefined) {
        highestValues[j] = {val: array[i], index: i}
        break
      }
      else if (array[i] > highestValues[j].val) {
        highestValues[j].val = array[i]
        highestValues[j].index = i
        break
      }
    }
  }
  highestValues.sort((a, b) => b.val - a.val)
  highestValues = highestValues.map(x => x.index)
  return highestValues
}

class App extends React.Component {

  styles = {
    editor: {
      width: "100%",
      height: "100%",
      fontSize: 18,
      fontFamily: "Sans-Serif",
      margin: "25px",
    }
  }

  // decorator = new CompositeDecorator([
  //   {
  //     strategy: findWords,
  //     component: AddSpan,
  //   },
  // ]);
  // AddSpan = (props) => {
  //   return <span {...props}>{props.children}</span>;
  // };
  // function

  constructor(props) {
    super(props)
    tf.setBackend("cpu") //using webgl backend causes slight freeze after every big matrix op
    console.log("✓ Loaded react")
    fetchWordVectors(loadProgress => this.setState({loadProgress}))
      .then(buffer => {
        this.setState({loadState: ["Loading word vectors", 1]})
        return loadWordVectors(buffer)
      }).then((wordData) => {
        this.setState({
          wordVectors: wordData[0],
          wordDict: wordData[1],
          vectorsLoaded: true,
          loadState: ["Ready", 3]
        })
      })
    this.state = {
      editorState: Draft.EditorState.createEmpty(),
      activeWordData: {text: "", x:0, y:0},
      similarWords: [],
      vectorsLoaded: false,
      wordDict: null,
      wordVectors: null,
      loadProgress: 0.0,
      loadState: ["Downloading word vectors", 0]
    }
  }

  handleKeyCommand(command, editorState) {
    const newState = Draft.RichUtils.handleKeyCommand(editorState, command);
    if (newState) {
      this.onChange(newState);
      return 'handled';
    }
    return 'not-handled';
  }

  //su eyr ad
  onChange(editorState) {
    this.setState((currentState, props) => {
      let activeWord = this.getActiveWord(editorState)
      let lastChange = editorState.getLastChangeType()

      let wasCharacterEdit = lastChange == 'insert-characters' || lastChange == 'backspace-character'
      let wasSelectionChange = currentState.editorState.getCurrentContent() == editorState.getCurrentContent()
      let isNewWord = activeWord != currentState.activeWordData.text
      let selectionExists = window.getSelection().rangeCount !=0
      let vectorAvailible = this.state.vectorsLoaded && activeWord in this.state.wordDict.fromWord

      if((wasCharacterEdit || wasSelectionChange) && isNewWord && selectionExists && vectorAvailible) {
        let cursor = window.getSelection().getRangeAt(0).getBoundingClientRect()
        let similarWords = this.getWordSuggestions(activeWord)
        return {
          editorState,
          activeWordData: {text: activeWord, x: cursor.x, y: cursor.y + cursor.height},
          similarWords
        }
      }
      else if (!isNewWord) {
        return {editorState}
      }
      else {
        return {
          editorState,
          activeWordData: {...currentState.activeWordData, text: activeWord},
          similarWords: [],
        }
      }
    })
  }

  getActiveWord(editorState) {
      let selectionState = editorState.getSelection();
      let anchorKey = selectionState.getAnchorKey();
      let currentContent = editorState.getCurrentContent();
      let currentContentBlock = currentContent.getBlockForKey(anchorKey);
      let text = currentContentBlock.getText()
      let start = selectionState.getStartOffset();
      let wordStart = text.lastIndexOf(" ", start-1)
      let wordEnd = text.indexOf(" ", start-1)
      if (wordStart == -1) wordStart = 0
      if (wordEnd == -1) wordEnd = text.length
      let word = text.substring(wordStart, wordEnd).trim().toLowerCase()
      return word
  }

  getWordSuggestions(word) {
    //make word vector
    let wordIndex = this.state.wordDict.fromWord[word]
    let wordVectorsBuffer = this.state.wordVectors.buffer()
    let vectorLength = this.state.wordVectors.shape[1]
    let vector = tf.buffer([vectorLength, 1], "float32")
    for (var idx = 0; idx < this.state.wordVectors.shape[1]; idx++) {
      vector.set(wordVectorsBuffer.get(wordIndex, idx), idx, 0)
    }
    vector = vector.toTensor()
    //get similarities
    let similarities = tf.matMul(this.state.wordVectors, vector)
    let similarityData = similarities.dataSync()
    similarities.dispose()
    vector.dispose()
    let similarWordIndexes = getHighestKIndices(similarityData, 10)
    let similarWords = similarWordIndexes.map(x => this.state.wordDict.fromIdx[x])
    return similarWords
  }

  render() {
    return (
      <div style={{height: "100%", width: "100%"}}>
        <StatusBar state={this.state.loadState} progress={this.state.loadProgress} loaded={this.state.vectorsLoaded} />
        <div style={this.styles.editor}>
          <Draft.Editor
            editorState={this.state.editorState}
            onChange={this.onChange.bind(this)}
            handleKeyCommand={this.handleKeyCommand.bind(this)}
            style={{height: "10px"}}
          />
          <Dropdown similarWords={this.state.similarWords} activeWordData={this.state.activeWordData} />
        </div>
      </div>
    )
  }
}

class Dropdown extends React.Component {

  style = {
    dropdown: {
      borderRadius: "7px",
      boxShadow: "0px 1px 4px #ccc",
      overflow: "hidden",
    },
    list: {
      listStyle: "none",
      padding: 0,
      margin: 0,
      fontSize: 15,
      boxSizing: "border-box",
    },
    item: {
      marginBottom: "-1px",
      borderBottom: "1px solid #ccc",
      padding: "2px",
      paddingLeft: "4px",
      paddingRight: "4px",
    }
  }

  constructor(props) {
    super(props)
  }

  render() {
    return (
      <div style={{position: "absolute", top: this.props.activeWordData.y, left: this.props.activeWordData.x, ...this.style.dropdown}}>
        <ul style={this.style.list}>
        {this.props.similarWords.map((word) =>
          <div key={word}>
            <li style={this.style.item}>{word}</li>
          </div>
        )}
        </ul>
      </div>
    )
  }
}

class StatusBar extends React.Component {

  style = {
    bar: {
      backgroundColor: "#222",
      fontFamily: "Sans-Serif",
      height: "40px",
      width: "100%",
      color: "white",
      display: "flex",
      alignItems: "center",
      paddingLeft: "25px",
    }
  }

  constructor(props) {
    super(props)
  }
  render () {
    let doShowProgress = this.props.state[1] == 0
    let progress = "(" + Math.round(this.props.progress * 100).toString() + "%)"
    return (
    <div style={this.style.bar}>
      <div>{this.props.state[0] + " "} {doShowProgress && progress}</div>
    </div>
    )
  }
}

ReactDOM.render(<App />, document.getElementById("root"))
