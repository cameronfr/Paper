// Cameron Franz, September 2018.
// TODO: novelty, 200ms wait before search to reduce lag, aesthetics, cleaning up word results.

const vectorURL = "https://storage.googleapis.com/mlstorage-cloud/Data/glove.6B.50d.txt.zip"

//fetch doesn't have progress, file too big to not have indicator ==> xhr ugliness.
function fetchWordVectors(onProgress) {
  return new Promise((resolve, reject) => {
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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function loadWordVectors(zipArrayBuffer, onProgress) {
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

    if (idx % Math.floor(lines.length / 100) == 0) {
      onProgress(parseInt(idx)/lines.length)
      await sleep(1) //react needs a frame to render or something like that
    }
  }

  console.log("✓ Loaded word vector matrix of shape " + wordVectors.shape)
  wordVectors = wordVectors.toTensor()
  return [wordVectors, wordDict]

}

function getHighestKIndices(array, k) {
  k = k + 1
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
  highestValues = highestValues.slice(1).map(x => x.index)
  return highestValues
}

class App extends React.Component {

  styles = {
    main: {
      height: "100%",
      width: "100%",
    },
    editor: {
      height: "100%", //no flexbox support for Draft.js editor; as a component doesn't respond well to css.
      marginTop: "-40px",
      padding: "25px",
      paddingTop: "65px",
      width: "100%",
      fontSize: 18,
      fontFamily: "Sans-Serif",
      boxSizing: "border-box",
    }
  }

  //"absolute" is positioned relative to closet parent with "relative", default position is static
  createSuggestionsDecorator(blockKey, start, end, similarWords) {
    let AddSpan = (props) => {
      return (<span id="activeWord" style={{position: "relative"}}>
        {props.children}
      </span>)
    };
    let findWords = (contentBlock, callBack, contentState) => {
      if (contentBlock.getKey() == blockKey) {
        callBack(start, end)
      }
    }
    let decorator = new Draft.CompositeDecorator([
      {
        strategy: findWords,
        component: AddSpan,
      },
    ]);
    return decorator
  }

  constructor(props) {
    super(props)
    localforage.setDriver(localforage.INDEXEDDB)
    tf.setBackend("cpu") //using webgl backend causes slight freeze after every big matrix op
    console.log("✓ Loaded react")
    let onProgress = loadProgress => {this.setState({loadProgress})}

    fetchWordVectors(onProgress)
      .then(buffer => {
        this.setState({loadState: ["Loading word vectors", 1], loadProgress: 0})
        return loadWordVectors(buffer, onProgress)
      }).then((wordData) => {
        this.setState({
          wordVectors: wordData[0],
          wordDict: wordData[1],
          vectorsLoaded: true,
          loadState: ["Ready", 3]
        })
      })

    let savedEditor = localStorage.getItem("savedEditor")
    if (savedEditor == null) {
      console.log("✓ Creating empty Draft editor")
      var newEditor = Draft.EditorState.createEmpty()
    } else {
      console.log("✓ Found saved Draft editor")
      var newEditor = Draft.EditorState.createWithContent(Draft.convertFromRaw(JSON.parse(savedEditor)))
    }

    this.state = {
      editorState: newEditor,
      activeWord: {text: "", blockKey: "", pos: null},
      similarWords: [],
      vectorsLoaded: false,
      wordDict: null,
      wordVectors: null,
      loadProgress: 0.0,
      loadState: ["Downloading word vectors", 0],
      lastChange: Date.now(),
      lastChangeTimeout: null,
    }

    let save = () => {
      let rawEditorState = JSON.stringify(Draft.convertToRaw(this.state.editorState.getCurrentContent()))
      localStorage.setItem("savedEditor", rawEditorState)
    }
    setInterval(save, 2000)
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
      let word = this.getActiveWord(editorState)
      let wordText = word.text
      let lastChangeType = editorState.getLastChangeType()

      //force minimum time diff for calculation when backspacing.
      let lastChangeTimeDiff = Date.now() - this.state.lastChange
      let wasLongDelay = lastChangeTimeDiff > 20 || lastChangeType != 'backspace-character'
      let lastChangeTimeout
      if (!wasLongDelay) {
        let updateEditor = () => {this.onChange(this.state.editorState)}
        lastChangeTimeout = setTimeout(updateEditor, 20)
      }
      clearTimeout(this.state.lastChangeTimeout)

      let wasCharacterEdit = lastChangeType == 'insert-characters' || lastChangeType == 'backspace-character'
      let wasSelectionChange = currentState.editorState.getCurrentContent() == editorState.getCurrentContent()
      let isNewWord = wordText != currentState.activeWord.text || word.blockKey != currentState.activeWord.blockKey
        || word.wordStart !=currentState.activeWord.wordStart
      let selectionExists = window.getSelection().rangeCount !=0
      let vectorAvailible = this.state.vectorsLoaded && wordText in this.state.wordDict.fromWord

      if((wasCharacterEdit || wasSelectionChange) && isNewWord && selectionExists && vectorAvailible && wasLongDelay) {
        let cursor = window.getSelection().getRangeAt(0).getBoundingClientRect()
        let similarWords = this.getWordSuggestions(wordText)
        let wordDecorator = this.createSuggestionsDecorator(word.blockKey, word.wordStart, word.wordEnd, similarWords)
        return {
          editorState: Draft.EditorState.set(editorState, {decorator: wordDecorator}),
          activeWord: word,
          similarWords,
          lastChange: Date.now(),
          lastChangeTimeout
        }
      }
      else if (!isNewWord) {
        return {editorState}
      }
      else {
        return {
          editorState: Draft.EditorState.set(editorState, {decorator: null}),
          similarWords: [],
          lastChange: Date.now(),
          lastChangeTimeout
        }
      }
    })
  }

  getActiveWord(editorState) {
      let selectionState = editorState.getSelection()
      let blockKey = selectionState.getAnchorKey()
      let currentContent = editorState.getCurrentContent();
      let currentContentBlock = currentContent.getBlockForKey(blockKey);
      let text = currentContentBlock.getText()
      let start = selectionState.getStartOffset()
      let end = selectionState.getEndOffset()
      let wordStart = text.lastIndexOf(" ", start-1)
      let wordEnd = text.indexOf(" ", start-1)
      let word
      if (start != end) {
        word = null
      }
      else {
        if (wordStart == -1) wordStart = 0
        if (wordEnd == -1) wordEnd = text.length
        word = text.substring(wordStart, wordEnd).trim().toLowerCase()
      }
      return {text: word, blockKey, wordStart, wordEnd}
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

  componentDidUpdate() {
    let currentWord = document.getElementById("activeWord")
    let lastPos = this.state.activeWord.pos
    if (currentWord) {
      let pos = currentWord.getBoundingClientRect()
      if (lastPos == null || (pos.x != lastPos.x && pos.y != lastPos.y)) {
        this.setState({activeWord: {...this.state.activeWord, pos}})
      }
    }
    else if (!currentWord && lastPos != null) {
        this.setState({activeWord: {...this.state.activeWord, pos: null}})
    }
  }

  render() {
    let doRenderDropdown = this.state.activeWord.pos != null
    if (doRenderDropdown) {
      var x = this.state.activeWord.pos.x
      var  y = this.state.activeWord.pos.y + 23
    }
    return (
      <div style={this.styles.main}>
        <StatusBar state={this.state.loadState} progress={this.state.loadProgress} loaded={this.state.vectorsLoaded} />
        <div style={this.styles.editor}>
          <Draft.Editor
            editorState={this.state.editorState}
            onChange={this.onChange.bind(this)}
            handleKeyCommand={this.handleKeyCommand.bind(this)}
            style={{}}
          />
        </div>
        {doRenderDropdown && <Dropdown similarWords={this.state.similarWords} position={{x, y}}/>}
      </div>
    )
  }
}

class Dropdown extends React.Component {

  style = {
    dropdown: {
      fontFamily: "Sans-Serif",
      borderRadius: "7px",
      boxShadow: "0px 1px 4px #ccc",
      overflow: "hidden",
      position: "absolute",
      backgroundColor: "white",
      zIndex: 100,
      MozUserSelect:"none",
      WebkitUserSelect:"none",
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
      <div style={{top: this.props.position.y, left: this.props.position.x, ...this.style.dropdown}}>
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
      boxSizing: "border-box",
    }
  }

  constructor(props) {
    super(props)
  }
  render () {
    let doShowProgress = this.props.state[1] == 0 || this.props.state[1] == 1
    let progress = "(" + Math.round(this.props.progress * 100).toString() + "%)"
    return (
      <div style={this.style.bar}>
        <div>{this.props.state[0] + " "} {doShowProgress && progress}</div>
      </div>
    )
  }
}

ReactDOM.render(<App />, document.getElementById("root"))
