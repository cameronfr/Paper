// Cameron Franz, September 2018.
// TODO: novelty, 200ms wait before search to reduce lag, aesthetics, cleaning up word results.

const vectorURL = "https://storage.googleapis.com/mlstorage-cloud/Data/glove.6B.50d.txt.zip"

async function loadWordVectors() {
  localforage.setDriver(localforage.INDEXEDDB)
  let zip = await localforage.getItem('vectorZip')
  if(zip == null) {
    zip = await fetch(vectorURL).then(res => res.arrayBuffer())
    await localforage.setItem('vectorZip', zip)
    console.log("✓ Downloaded and stored vector file")
  } else {
    console.log("✓ Found stored vector file")
  }
  let text = await JSZip.loadAsync(zip).then(res => res.files["glove.6B.50d.txt"].async("string"))
      .catch(err => console.log("✗ Error fetching and unzipping vector file:\n" + err))

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

  vectorsLoaded = false
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
    loadWordVectors().then((wordData) => {
      this.wordVectors = wordData[0]
      this.wordDict = wordData[1]
      this.vectorsLoaded = true
    })
    this.state = {
      editorState: Draft.EditorState.createEmpty(),
      activeWordData: {text: "", x:0, y:0},
      similarWords: [],
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
      let vectorAvailible = this.vectorsLoaded && activeWord in this.wordDict.fromWord

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
    let wordIndex = this.wordDict.fromWord[word]
    let wordVectorsBuffer = this.wordVectors.buffer()
    let vectorLength = this.wordVectors.shape[1]
    let vector = tf.buffer([vectorLength, 1], "float32")
    for (var idx = 0; idx < this.wordVectors.shape[1]; idx++) {
      vector.set(wordVectorsBuffer.get(wordIndex, idx), idx, 0)
    }
    vector = vector.toTensor()
    //get similarities
    let similarities = tf.matMul(this.wordVectors, vector)
    let similarityData = similarities.dataSync()
    similarities.dispose()
    vector.dispose()
    let similarWordIndexes = getHighestKIndices(similarityData, 10)
    let similarWords = similarWordIndexes.map(x => this.wordDict.fromIdx[x])
    return similarWords
  }

  render() {
    return (
      <div style={this.styles.editor}>
        <Draft.Editor
          editorState={this.state.editorState}
          onChange={this.onChange.bind(this)}
          handleKeyCommand={this.handleKeyCommand.bind(this)}
          style={{height: "10px"}}
        />
        <Dropdown similarWords={this.state.similarWords} activeWordData={this.state.activeWordData} />
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

ReactDOM.render(<App />, document.getElementById("root"))
